# -*- coding: utf-8 -*-
"""
Membership Inference Attack (MIA) - OPTION B ("real-world attacker" scenario).

The target models were trained on ANONYMIZED text (bert_score_pred.py:
train uses column 1 + gender-word masking). Here, the ATTACK instead
uses the ORIGINAL, non-anonymized text for BOTH members and non-members,
simulating an attacker who only has access to a candidate's real CV
(never anonymized, since anonymization only exists inside the training
pipeline).

Fully self-contained via auto-discovery -- does NOT require ANON_ROOT
or a CONFIGURATIONS list, and does NOT require modifying
bert_score_pred.py. It only needs what that script already saves:

  models/<model_name>/<tag>/model_biased.pt           (the trained model)
  models/<model_name>/<tag>/test_text_processed.npy   (already RAW test text)
  models/train_subset_indices.npy                     (shared fixed train indices)

The RAW train text (members) is reconstructed on the fly from the
ORIGINAL FairCVdb.npy ('Bios Train', column 0), indexed with the same
train_subset_indices.npy used to train each model -- guaranteeing the
exact same candidates, with the exact same (non-anonymized) text.

For each configuration:
  1. Loads the model checkpoint.
  2. Builds members = raw train text (FairCVdb.npy, column 0, indexed by
     train_subset_indices.npy) and non-members = raw test text (already
     saved as test_text_processed.npy).
  3. Computes, per sample: squared prediction error + L2 norm of the
     gradient of that sample's loss w.r.t. the model's trainable
     parameters (l1, l2, l3).
  4. Trains a small attack MLP on a balanced subset (half of each class
     to train the attack, half to evaluate it), reports AUC-ROC.

RESUME SUPPORT: before running, the script reads the results CSV (if it
already exists) and skips any configuration whose tag is already present
there, so re-running only processes the configurations still missing.
"""

import os
import re
import gc
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizer, set_seed

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# Preprocessing / Model (matching bert_score_pred.py)
# ============================================================
def preprocessing(example):
    example = str(example).lower()
    return re.sub(r'[^\w\s]', '', example)


class BERTRegression(torch.nn.Module):
    def __init__(self, model_name, hidden_size=300, comp_size=7, drop_rate=0.3,
                 pool_embeddings=True):
        super(BERTRegression, self).__init__()
        if 'roberta' in model_name:
            self.backbone = RobertaModel.from_pretrained(model_name)
        else:
            self.backbone = BertModel.from_pretrained(model_name)
        for _, param in self.backbone.named_parameters():
            param.requires_grad = False
        self.l1 = torch.nn.Linear(self.backbone.config.hidden_size, hidden_size)
        self.d1 = torch.nn.Dropout(drop_rate)
        self.l2 = torch.nn.Linear(hidden_size, 20)
        self.d2 = torch.nn.Dropout(drop_rate)
        self.l3 = torch.nn.Linear(20 + comp_size, 1)
        self.pooling = pool_embeddings

    def average_pooling(self, model_output, mask):
        token_emb = model_output[0]
        input_mask_expanded = mask.unsqueeze(-1).expand(token_emb.size()).float()
        sentence_emb = torch.sum(token_emb * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)
        return F.normalize(sentence_emb, p=2, dim=1)

    def forward_with_activations(self, ids, comp, mask=None, token_type_ids=None):
        sentence_emb = self.backbone(ids, attention_mask=mask, token_type_ids=token_type_ids)
        if self.pooling:
            hidden_1 = self.l1(self.average_pooling(sentence_emb, mask))
        else:
            hidden_1 = self.l1(sentence_emb[1])
        hidden_2 = self.d1(torch.sigmoid(hidden_1))
        hidden_3 = self.l2(hidden_2)
        hidden_4 = self.d2(torch.sigmoid(hidden_3))
        hidden_5 = torch.cat((hidden_4, comp), dim=1)
        outputs = torch.sigmoid(self.l3(hidden_5))
        return outputs.squeeze(-1)


def compute_error_and_gradient_features(model, text_array, comp, targets, tokenizer, max_len, device):
    """For each sample: squared prediction error + L2 norm of the
    gradient of that sample's loss w.r.t. the model's trainable
    parameters (l1, l2, l3). One sample at a time (per-sample gradients
    cannot be batched trivially)."""
    model.eval()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    criterion = torch.nn.MSELoss()

    errors = []
    grad_norms = []
    n = len(text_array)

    for i in range(n):
        text = str(text_array[i])
        text = " ".join(text.split())
        inputs = tokenizer.encode_plus(
            text, None, add_special_tokens=True, max_length=max_len,
            padding='max_length', return_token_type_ids=True, truncation=True
        )
        ids = torch.tensor([inputs['input_ids']], dtype=torch.long).to(device)
        mask = torch.tensor([inputs['attention_mask']], dtype=torch.long).to(device)
        token_type_ids = torch.tensor([inputs['token_type_ids']], dtype=torch.long).to(device)
        comp_i = torch.tensor([comp[i, :]], dtype=torch.float).to(device)
        target_i = torch.tensor([targets[i]], dtype=torch.float).to(device)

        model.zero_grad()
        output_i = model.forward_with_activations(ids, comp_i, mask, token_type_ids)
        loss = criterion(output_i, target_i)

        error_val = loss.item()
        loss.backward()

        total_norm_sq = 0.0
        for p in trainable_params:
            if p.grad is not None:
                total_norm_sq += p.grad.detach().pow(2).sum().item()
        grad_norm = total_norm_sq ** 0.5

        errors.append(error_val)
        grad_norms.append(grad_norm)
        model.zero_grad()

        if (i + 1) % 100 == 0 or (i + 1) == n:
            print('    processed {}/{} samples'.format(i + 1, n))

    return np.array(errors).reshape(-1, 1), np.array(grad_norms).reshape(-1, 1)


# ============================================================
# Small attack classifier (2 features: error + grad norm)
# ============================================================
class SmallAttackMLP(torch.nn.Module):
    def __init__(self, input_dim):
        super(SmallAttackMLP, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_small_attack_mlp(X_train, y_train, X_eval, y_eval, device,
                            epochs=150, batch_size=32, lr=1e-3, weight_decay=1e-3, seed=123):
    torch.manual_seed(seed)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_eval_t = torch.tensor(X_eval, dtype=torch.float32).to(device)

    attack_model = SmallAttackMLP(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    n_samples = X_train_t.shape[0]

    for epoch in range(epochs):
        attack_model.train()
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            logits = attack_model(xb)
            batch_loss = criterion(logits, yb)
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item() * xb.size(0)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print('    Attack MLP - Epoch {}/{}: loss = {:.5f}'.format(
                epoch + 1, epochs, epoch_loss / n_samples))

    attack_model.eval()
    with torch.no_grad():
        train_probs = torch.sigmoid(attack_model(X_train_t)).cpu().numpy()
        eval_probs = torch.sigmoid(attack_model(X_eval_t)).cpu().numpy()

    train_acc = ((train_probs >= 0.5).astype(float) == y_train).mean()
    eval_acc = ((eval_probs >= 0.5).astype(float) == y_eval).mean()
    auc = roc_auc_score(y_eval, eval_probs)

    del attack_model, X_train_t, y_train_t, X_eval_t
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return train_acc, eval_acc, auc


# ============================================================
# ================  CONFIGURATION  ============================
# ============================================================
MODELS_ROOT = r'D:\Gonzalo\Compartir\NLP\NLPCV\MIA\data\models'
ORIGINAL_DATA_FILE = r'D:\Gonzalo\Compartir\NLP\NLPEGNA\code\data\FairCVdb.npy'

BERT = True
MAX_LEN = 64
RANDOM_SEED = 123
BIAS = True

model_name = 'bert-base-uncased' if BERT else 'roberta-base'

RESULTS_PATH = os.path.join(MODELS_ROOT, 'mia_optionB_error_grad_resultsroberta.csv')


def discover_configurations(models_root, model_name, bias):
    """Auto-discovers configurations by scanning models_root/model_name/
    for subfolders that already contain a trained model checkpoint AND
    the raw test text saved by bert_score_pred.py (test_text_processed.npy).
    No dependency on ANON_ROOT or a CONFIGURATIONS list."""
    base_dir = os.path.join(models_root, model_name)
    model_filename = 'model_biased.pt' if bias else 'model_unbiased.pt'
    configs = []
    if not os.path.isdir(base_dir):
        print('WARNING: {} does not exist -- no configurations found.'.format(base_dir))
        return configs
    for tag in sorted(os.listdir(base_dir)):
        tag_dir = os.path.join(base_dir, tag)
        if not os.path.isdir(tag_dir):
            continue
        configs.append({
            'tag': tag,
            'model_path': os.path.join(tag_dir, model_filename),
            'test_text_path': os.path.join(tag_dir, 'test_text_processed.npy'),
        })
    return configs


def load_completed_configs(results_path):
    """Reads the results CSV (if it exists) and returns the set of
    'config' tags that already have a row -- i.e. configurations that
    do not need to be re-run."""
    if not os.path.exists(results_path):
        print('No existing results file at {} -- starting fresh.'.format(results_path))
        return set()
    try:
        df = pd.read_csv(results_path)
        if 'config' not in df.columns:
            print('WARNING: {} exists but has no "config" column -- treating as empty.'.format(results_path))
            return set()
        completed = set(df['config'].astype(str).unique())
        print('Found {} already-completed configuration(s) in {}.'.format(len(completed), results_path))
        return completed
    except Exception as e:
        print('WARNING: could not read {} ({}) -- treating as empty.'.format(results_path, e))
        return set()


# ============================================================
# Run
# ============================================================
if __name__ == '__main__':

    set_seed(RANDOM_SEED)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Using device: {}'.format(device))

    # --- Shared data: original (raw) FairCVdb + fixed train subset ---
    fairCV = np.load(ORIGINAL_DATA_FILE, allow_pickle=True).item()
    profiles_train = fairCV['Profiles Train']
    profiles_test = fairCV['Profiles Test']
    bios_train_original_full = fairCV['Bios Train'][:, 0]  # RAW train text, all 19200

    if BIAS:
        scores_train_full = fairCV['Biased Labels Train (Gender)']
        scores_test_full = fairCV['Biased Labels Test (Gender)']
    else:
        scores_train_full = fairCV['Blind Labels Train']
        scores_test_full = fairCV['Blind Labels Test']

    indices_path = os.path.join(MODELS_ROOT, 'train_subset_indices.npy')
    if not os.path.exists(indices_path):
        raise FileNotFoundError(
            '{} not found -- run bert_score_pred.py first (it generates '
            'this file).'.format(indices_path)
        )
    train_subset_idx = np.load(indices_path)
    print('Loaded fixed train subset: {} indices from {}'.format(len(train_subset_idx), indices_path))

    # Raw train text + comp + scores, for the SAME candidates used to
    # train every model, but WITHOUT anonymization or gender-masking.
    bios_train_raw_subset = bios_train_original_full[train_subset_idx]
    bios_train_raw_subset = np.array([preprocessing(t) for t in bios_train_raw_subset])
    comp_train_subset = profiles_train[train_subset_idx, 4:11]
    scores_train_subset = scores_train_full[train_subset_idx]

    comp_test_full = profiles_test[:, 4:11]

    if 'roberta' in model_name:
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)

    results_ok = []
    missing_files = []
    failed = []

    all_configurations = discover_configurations(MODELS_ROOT, model_name, BIAS)
    print('Discovered {} configuration(s) under {}:'.format(
        len(all_configurations), os.path.join(MODELS_ROOT, model_name)))
    for c in all_configurations:
        print('  - {}'.format(c['tag']))

    # --- Resume support: skip configs already present in the results CSV ---
    completed_tags = load_completed_configs(RESULTS_PATH)
    configurations = [c for c in all_configurations if c['tag'] not in completed_tags]
    skipped_already_done = [c['tag'] for c in all_configurations if c['tag'] in completed_tags]

    if skipped_already_done:
        print('\nSkipping {} configuration(s) already present in the results CSV:'.format(
            len(skipped_already_done)))
        for tag in skipped_already_done:
            print('  - {} (already done)'.format(tag))

    print('\n{} configuration(s) left to run.'.format(len(configurations)))

    for cfg in configurations:
        tag = cfg['tag']
        print('\n' + '=' * 70)
        print('CONFIGURATION: {} (Option B -- raw/raw)'.format(tag))
        print('=' * 70)

        model_path = cfg['model_path']
        test_text_path = cfg['test_text_path']

        missing_here = []
        if not os.path.exists(model_path):
            missing_here.append(model_path)
        if not os.path.exists(test_text_path):
            missing_here.append(test_text_path)

        if missing_here:
            print('  [SKIPPED] Missing file(s):')
            for f in missing_here:
                print('    - {}'.format(f))
            missing_files.append({'config': tag, 'missing_paths': missing_here})
            continue

        model = None

        try:
            # Non-members: already-saved RAW test text (column 0 +
            # preprocessing only, as saved by bert_score_pred.py)
            test_text_saved = np.load(test_text_path, allow_pickle=True)

            assert len(test_text_saved) == len(profiles_test), (
                f"Mismatch: {len(test_text_saved)} saved test texts vs "
                f"{len(profiles_test)} test profiles."
            )

            print('Members (train, RAW): {} samples'.format(len(bios_train_raw_subset)))
            print('Non-members (test, RAW): {} samples'.format(len(test_text_saved)))

            model = BERTRegression(model_name, pool_embeddings=True)
            state_dict = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()

            print('Computing error + gradient norm for TRAIN (members, raw)...')
            train_error, train_grad = compute_error_and_gradient_features(
                model, bios_train_raw_subset, comp_train_subset, scores_train_subset,
                tokenizer, MAX_LEN, device)

            print('Computing error + gradient norm for TEST (non-members, raw)...')
            test_error, test_grad = compute_error_and_gradient_features(
                model, test_text_saved, comp_test_full, scores_test_full,
                tokenizer, MAX_LEN, device)

            train_rmse = np.sqrt(train_error.mean())
            test_rmse = np.sqrt(test_error.mean())
            print('  Train RMSE: {:.5f} | Test RMSE: {:.5f}'.format(train_rmse, test_rmse))

            train_feats = np.concatenate([train_error, train_grad], axis=1)
            test_feats_full = np.concatenate([test_error, test_grad], axis=1)

            # Cap per class = number of train candidates used for this
            # model, so classes are balanced going into the attack.
            max_samples_per_class = len(train_feats)
            rng = np.random.RandomState(RANDOM_SEED)

            test_idx_all = np.arange(len(test_feats_full))
            if len(test_idx_all) > max_samples_per_class:
                test_idx_all = rng.choice(test_idx_all, size=max_samples_per_class, replace=False)
            test_feats = test_feats_full[test_idx_all]

            print('  Using {} samples per class'.format(max_samples_per_class))

            train_labels = np.ones(len(train_feats))
            test_labels = np.zeros(len(test_feats))

            N_ATTACK_TRAIN_PER_CLASS = max_samples_per_class // 2
            train_idx_all2 = np.arange(len(train_feats))
            test_idx_all2 = np.arange(len(test_feats))

            train_idx_for_attack = rng.choice(train_idx_all2, size=N_ATTACK_TRAIN_PER_CLASS, replace=False)
            test_idx_for_attack = rng.choice(test_idx_all2, size=N_ATTACK_TRAIN_PER_CLASS, replace=False)

            train_idx_remaining = np.setdiff1d(train_idx_all2, train_idx_for_attack)
            test_idx_remaining = np.setdiff1d(test_idx_all2, test_idx_for_attack)

            X_attack_train = np.concatenate([train_feats[train_idx_for_attack],
                                              test_feats[test_idx_for_attack]], axis=0)
            y_attack_train = np.concatenate([train_labels[train_idx_for_attack],
                                              test_labels[test_idx_for_attack]], axis=0)

            X_attack_eval = np.concatenate([train_feats[train_idx_remaining],
                                             test_feats[test_idx_remaining]], axis=0)
            y_attack_eval = np.concatenate([train_labels[train_idx_remaining],
                                             test_labels[test_idx_remaining]], axis=0)

            print('Attack train set: {} samples | Attack eval set: {} samples'.format(
                len(X_attack_train), len(X_attack_eval)))

            scaler = StandardScaler()
            X_attack_train_s = scaler.fit_transform(X_attack_train)
            X_attack_eval_s = scaler.transform(X_attack_eval)

            train_acc, eval_acc, auc = train_small_attack_mlp(
                X_attack_train_s, y_attack_train, X_attack_eval_s, y_attack_eval,
                device=device, epochs=150, batch_size=32, seed=RANDOM_SEED
            )

            print('  Attack AUC-ROC (held-out): {:.4f}'.format(auc))

            anonymizer_name, entity_name = (tag.rsplit('_', 1) if '_' in tag else (tag, ''))
            row = {
                'anonymizer': anonymizer_name, 'entity': entity_name, 'config': tag,
                'model_path': model_path,
                'n_used_per_class': max_samples_per_class,
                'train_rmse': train_rmse, 'test_rmse': test_rmse,
                'attack_train_acc': train_acc, 'attack_eval_acc': eval_acc,
                'attack_auc': auc
            }
            results_ok.append(row)

            row_df = pd.DataFrame([row])
            if os.path.exists(RESULTS_PATH):
                row_df.to_csv(RESULTS_PATH, mode='a', header=False, index=False)
            else:
                row_df.to_csv(RESULTS_PATH, mode='w', header=True, index=False)
            print('  Result saved to: {}'.format(RESULTS_PATH))

        except Exception as e:
            print('  !! FAILED: {}'.format(e))
            traceback.print_exc()
            failed.append({'config': tag, 'error': str(e)})

        finally:
            print('  Cleaning up memory before next configuration...')
            if model is not None:
                model.to('cpu')
                del model
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print('  Memory cleanup done.')

    # ============================================================
    # Final summary
    # ============================================================
    print('\n' + '=' * 70)
    print('FINAL SUMMARY (Option B -- raw/raw)')
    print('=' * 70)

    print('\n✅ Completed successfully this run ({}):'.format(len(results_ok)))
    for item in results_ok:
        print('  - {}: AUC = {:.4f} (train RMSE={:.4f}, test RMSE={:.4f})'.format(
            item['config'], item['attack_auc'], item['train_rmse'], item['test_rmse']))

    print('\n⏭️  Skipped -- already done in a previous run ({}):'.format(len(skipped_already_done)))
    for tag in skipped_already_done:
        print('  - {}'.format(tag))

    print('\n⚠️  Skipped (missing files) ({}):'.format(len(missing_files)))
    for item in missing_files:
        print('  - {}:'.format(item['config']))
        for p in item['missing_paths']:
            print('      missing: {}'.format(p))

    if failed:
        print('\n❌ Failed with an error ({}):'.format(len(failed)))
        for item in failed:
            print('  - {}: {}'.format(item['config'], item['error']))

    # Combine this run's results with whatever was already in the CSV
    # (if any) so the final AUC summary reflects everything completed so far.
    if os.path.exists(RESULTS_PATH):
        try:
            full_results_df = pd.read_csv(RESULTS_PATH)
        except Exception:
            full_results_df = pd.DataFrame(results_ok)
    else:
        full_results_df = pd.DataFrame(results_ok)

    if not full_results_df.empty:
        print('\n' + '=' * 70)
        print('AVERAGE AUC PER ANONYMIZER (Option B, all completed configurations)')
        print('=' * 70)
        avg_per_anonymizer = full_results_df.groupby('anonymizer')['attack_auc'].agg(['mean', 'std', 'count'])
        for anonymizer, row in avg_per_anonymizer.iterrows():
            std_str = '{:.4f}'.format(row['std']) if row['count'] > 1 and not pd.isna(row['std']) else 'n/a'
            print('  {}: mean AUC = {:.4f} (std = {}, n = {})'.format(
                anonymizer, row['mean'], std_str, int(row['count'])))
        overall_mean = full_results_df['attack_auc'].mean()
        print('\n  Overall mean AUC across all completed configurations: {:.4f}'.format(overall_mean))
        summary_path = os.path.join(MODELS_ROOT, 'mia_optionB_error_grad_auc_summaryroberta.csv')
        avg_per_anonymizer.reset_index().to_csv(summary_path, index=False)
        print('  Summary saved to: {}'.format(summary_path))
    else:
        print('\nNo completed configurations to summarize.')