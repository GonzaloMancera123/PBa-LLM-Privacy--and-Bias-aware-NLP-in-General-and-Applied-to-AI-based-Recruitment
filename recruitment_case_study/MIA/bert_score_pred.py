# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 18:44:58 2023

@author: Alejandro

BATCH version: trains one model per (anonymizer, entity) configuration
listed in CONFIGURATIONS below. Same hyperparameters, same architecture,
and same tokenizer call (encode_plus, left UNCHANGED as requested) as
the original single-model script. Only the outer loop, file-existence
checks, and per-configuration memory cleanup were added.
"""
import time
import os
import re
import gc
import traceback

import pandas as pd
import numpy as np

import seaborn as sns

from transformers import BertTokenizer, BertModel, set_seed, get_linear_schedule_with_warmup
from transformers import RobertaModel, RobertaTokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW


# CUDA flag to allow more detailed error reports 
CUDA_LAUNCH_BLOCKING = 1

# Data preprocessing to remove punctuation
def preprocessing(example):
    example = example.lower()
    return re.sub(r'[^\w\s]','', example)

# Data preprocessing to mask proxy words
def remove_gendered_words(example, name):
    mask_token = '[MASK]'
    gendered = ['he', 'she', 'mr', 'ms', 'her', 'his','him','children',
                'family','husband']+ [name.lower()]
    example = example.lower().split(' ')
    example = ' '.join([w if w not in gendered else mask_token for w in example])
    
    return example

# Custom Dataset
class CustomDataset(Dataset):

    # Init routine, we save the dataset, the tokenizer and the context lenght
    def __init__(self, text_data, comp, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = text_data
        self.comp = comp
        self.max_len = max_len
    
    # Length of the dataset
    def __len__(self):
        return len(self.data)

    # Routine to draw examples
    def __getitem__(self, index):
        text = str(self.data.Text[index])
        text = " ".join(text.split())

        # Tokenize the data
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding = 'max_length',
            return_token_type_ids=True,
            truncation = True
        )
        
        # Extract token ids, mask and token type
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        # Return in a dict object the tokenized data, the competencies and target labels
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'comp': torch.tensor(self.comp[index,:], dtype = torch.float),
            'targets': torch.tensor(self.data.Label[index], dtype=torch.float)
        }

# We define a custom regression model for the scoring tool based on the BERT architecture
class BERTRegression(torch.nn.Module):
    
    # Init routine
    def __init__(self, model_name, hidden_size = 300, comp_size = 7, drop_rate = 0.3,
                 pool_embeddings = True):
        
        super(BERTRegression, self).__init__()
        
        # Select the Text Understanding Backbone
        if 'roberta' in model_name:
            self.backbone = RobertaModel.from_pretrained(model_name)
        else:
            self.backbone = BertModel.from_pretrained(model_name)
        
        # Freeze the backbone
        for _, param in self.backbone.named_parameters():
            param.requires_grad = False
        
        # Definition of the Fusion module
        self.l1 = torch.nn.Linear(self.backbone.config.hidden_size, hidden_size)
        self.d1 = torch.nn.Dropout(drop_rate)
        self.l2 = torch.nn.Linear(hidden_size, 20)
        self.d2 = torch.nn.Dropout(drop_rate)
        self.l3 = torch.nn.Linear(20 + comp_size, 1)
        
        # Use the pooling layer as text representation, or the output for the CLS token
        self.pooling = pool_embeddings
    
    # Forward routine
    def forward(self, ids, comp, mask = None, token_type_ids = None):
        
        # Extract text embeddings
        sentence_emb = self.backbone(ids, attention_mask = mask, token_type_ids = token_type_ids)
        if self.pooling:
            hidden_1 = self.l1(self.average_pooling(sentence_emb, mask))
        else:
            hidden_1 = self.l1(sentence_emb[1])
 
        hidden_2 = self.d1(torch.sigmoid(hidden_1))
        hidden_3 = self.l2(hidden_2)
        hidden_4 = self.d2(torch.sigmoid(hidden_3))
        
        # Concat latent representation with competencies and return the score
        hidden_5 = torch.cat((hidden_4, comp), dim = 1)
        outputs = torch.sigmoid(self.l3(hidden_5))
        
        return outputs, hidden_2
    
    # Average output embeddings
    def average_pooling(self, model_output, mask):
        
        # Extract the hidden states from the output
        token_emb = model_output[0]
        
        # Expand input mask and average embeddings considering the mask
        input_mask_expanded = mask.unsqueeze(-1).expand(token_emb.size()).float()
        sentence_emb = torch.sum(token_emb * input_mask_expanded, 1)/torch.clamp(input_mask_expanded.sum(1), min = 1e-9)
        
        # normalize embeddings
        sentence_emb = F.normalize(sentence_emb, p=2, dim = 1)
        
        return sentence_emb

# Training routine
def train(epoch, model, optimizer, scheduler, train_loader, device, loss):
    
    # Parameters
    running_loss = 0.0
    epoch_loss = 0.0
    
    # Set train mode and loop over dataset
    model.train()
    for ix, data in enumerate(train_loader, 0):
        
        # Get batch data
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        comp = data['comp'].to(device, dtype = torch.float)
        targets = data['targets'].to(device, dtype = torch.float)
        
        # Forward pass
        outputs,_ = model(ids, comp, mask, token_type_ids)
        
        # Set gradients at zero
        optimizer.zero_grad()
        
        # Compute batch loss, backward pass and update the weights
        batch_loss = torch.sqrt(loss(outputs.squeeze(), targets.squeeze()))
        batch_loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Save loss info and print if necessary
        total_batch_loss = batch_loss.item() * ids.size(0)
        running_loss += total_batch_loss
        epoch_loss += total_batch_loss
        if ix % 100 == 0:
            print('Epoch: {}-{}, Loss: {:.5f}'.format(epoch, ix, running_loss/(100*ids.size(0))))
            running_loss = 0.0
            
    # Print average epoch loss
    print('Epoch: {}, Train Loss: {:.5f}'.format(epoch, epoch_loss/len(train_loader.dataset)))
            
# Validation routine            
def validate(epoch, model, test_loader, device, loss):
    
    # Parameters
    fin_targets=[]
    fin_outputs=[]
    val_loss = 0.0
    
    # Set evaluation mode
    model.eval()
    
    # We don't want to store gradient tensors during validation
    with torch.no_grad():
        for _, data in enumerate(test_loader, 0):
            
            # Get batch data
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            comp = data['comp'].to(device, dtype = torch.float)
            targets = data['targets'].to(device, dtype = torch.float)
            
            # Forward pass
            outputs,_ = model(ids, comp, mask, token_type_ids)
            
            # Compute validation loss
            batch_loss = torch.sqrt(loss(outputs.squeeze(), targets.squeeze()))
            val_loss += batch_loss.item() * ids.size(0)
            
            # Store targets and predicted scores
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
            
    # Print validation loss        
    print('Epoch: {}, Val Loss: {:.5f}'.format(epoch, val_loss/len(test_loader.dataset)))
            
    return fin_outputs, fin_targets        


# ============================================================
# ================  EDIT THIS CONFIGURATION LIST  ============
# ============================================================
# Root directory changed to: D:\Gonzalo\Compartir\NLP\NLPCV\Anonimization
# Train file path is assumed to follow the same pattern as Test,
# with 'Train' instead of 'Test' in the folder name. Adjust if your
# actual folder layout differs for any anonymizer.
ANON_ROOT = r'D:\Gonzalo\Compartir\NLP\NLPCV\Anonimization'

CONFIGURATIONS = [
    # {'name': 'Flair', 'entity': 'LOC',
    #  'train_file': os.path.join(ANON_ROOT, 'Flair', 'Train', 'anonymized_fairCVLOC.npy'),
    #  'test_file': os.path.join(ANON_ROOT, 'Flair', 'Test', 'anonymized_fairCVLOC.npy')},

    # {'name': 'Flair', 'entity': 'PER',
    #  'train_file': os.path.join(ANON_ROOT, 'Flair', 'Train', 'anonymized_fairCVPER.npy'),
    #  'test_file': os.path.join(ANON_ROOT, 'Flair', 'Test', 'anonymized_fairCVPER.npy')},

    # {'name': 'Presidio', 'entity': 'LOC',
    #  'train_file': os.path.join(ANON_ROOT, 'Presidio', 'Train', 'anonymized_fairCVLOC.npy'),
    #  'test_file': os.path.join(ANON_ROOT, 'Presidio', 'Test', 'anonymized_fairCVLOC.npy')},

    # {'name': 'Presidio', 'entity': 'PER',
    #  'train_file': os.path.join(ANON_ROOT, 'Presidio', 'Train', 'anonymized_fairCVPER.npy'),
    #  'test_file': os.path.join(ANON_ROOT, 'Presidio', 'Test', 'anonymized_fairCVPER.npy')},

    # {'name': 'DeepPavlov', 'entity': 'LOC',
    #  'train_file': os.path.join(ANON_ROOT, 'DeepPavlov', 'Train', 'anonymized_fairCVLOC.npy'),
    #  'test_file': os.path.join(ANON_ROOT, 'DeepPavlov', 'Test', 'anonymized_fairCVLOC.npy')},

    # {'name': 'DeepPavlov', 'entity': 'PER',
    #  'train_file': os.path.join(ANON_ROOT, 'DeepPavlov', 'Train', 'anonymized_fairCVPersons.npy'),
    #  'test_file': os.path.join(ANON_ROOT, 'DeepPavlov', 'Test', 'anonymized_fairCVPER.npy')},

    # {'name': 'ChatGPT4MINI', 'entity': 'LOC',
    #  'train_file': os.path.join(ANON_ROOT, 'ChatGPT', 'ChatGPT4MINI', 'Train', 'anonymized_fairCVCHATGPTLOC4mini.npy'),
    #  'test_file': os.path.join(ANON_ROOT, 'ChatGPT', 'ChatGPT4MINI', 'Test', 'anonymized_fairCVCHATGPTLOC4mini.npy')},

    # {'name': 'ChatGPT4MINI', 'entity': 'PER',
    #  'train_file': os.path.join(ANON_ROOT, 'ChatGPT', 'ChatGPT4MINI', 'Train', 'anonymized_fairCVCHATGPTPER4mini.npy'),
    #  'test_file': os.path.join(ANON_ROOT, 'ChatGPT', 'ChatGPT4MINI', 'Test', 'anonymized_fairCVCHATGPTPER4mini.npy')},

    # {'name': 'ChatGPT35', 'entity': 'LOC',
    #  'train_file': os.path.join(ANON_ROOT, 'ChatGPT', 'ChatGPT35', 'Train', 'anonymized_fairCVCHATGPTLOC35.npy'),
    #  'test_file': os.path.join(ANON_ROOT, 'ChatGPT', 'ChatGPT35', 'Test', 'anonymized_fairCVCHATGPTLOC35.npy')},

    # {'name': 'ChatGPT35', 'entity': 'PER',
    #  'train_file': os.path.join(ANON_ROOT, 'ChatGPT', 'ChatGPT35', 'Train', 'anonymized_fairCVCHATGPTPER35.npy'),
    #  'test_file': os.path.join(ANON_ROOT, 'ChatGPT', 'ChatGPT35', 'Test', 'anonymized_fairCVCHATGPTPER35.npy')},

         {'name': 'QWEN3', 'entity': 'LOC',
     'train_file': os.path.join(ANON_ROOT, 'QWEN3', 'Train', 'anonymized_fairCVLOC.npy'),
     'test_file': os.path.join(ANON_ROOT, 'QWEN3', 'Test', 'anonymized_fairCVLOC.npy')},

    {'name': 'QWEN3', 'entity': 'PER',
     'train_file': os.path.join(ANON_ROOT, 'QWEN3', 'Train', 'anonymized_fairCVPER.npy'),
     'test_file': os.path.join(ANON_ROOT, 'QWEN3', 'Test', 'anonymized_fairCVPER.npy')},
]

# Original FairCVdb.npy is still needed for Profiles/Names/Labels, since
# the anonymized files only contain the raw text pairs (column 0 =
# original, column 1 = anonymized). ADJUST THIS PATH to wherever your
# original FairCVdb.npy actually lives.
ORIGINAL_DATA_FILE = r'D:\Gonzalo\Compartir\NLP\NLPEGNA\code\data\FairCVdb.npy'
MODELS_ROOT = r'D:\Gonzalo\Compartir\NLP\NLPCV\MIA\data\models'

# Learning parameters (same as your original single-model script)
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
EPOCHS = 200
LEARNING_RATE = 1e-03
MAX_LEN = 64
EPS = 1e-08
POOLING = True
BERT = True
BIAS = True


# ============================================================
# Run
# ============================================================
if __name__ == '__main__':

    set_seed(123)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Using device: {}'.format(device))

    model_name = 'bert-base-uncased' if BERT else 'roberta-base'

    # Load ORIGINAL FairCV data once (Profiles/Names/Labels are shared
    # across all anonymizers -- only the text changes)
    fairCV = np.load(ORIGINAL_DATA_FILE, allow_pickle=True).item()
    profiles_train = fairCV['Profiles Train']
    names_train = fairCV['Names Train']
    profiles_test = fairCV['Profiles Test']
    names_test = fairCV['Names Test']

    # ------------------------------------------------------------
    # Fixed train subset (4000 out of the full 19200), shared by ALL
    # configurations (Baseline + every anonymizer) so they all train on
    # the exact same underlying candidates -- only the text
    # anonymization differs between configurations. Generated once and
    # reused on every subsequent run (train_baseline.py loads the same
    # file, so make sure this script runs first, or that file, whichever
    # is run first creates it).
    # ------------------------------------------------------------
    N_TRAIN_SUBSET = 400
    TRAIN_SUBSET_SEED = 123

    indices_path = os.path.join(MODELS_ROOT, 'train_subset_indices.npy')
    if os.path.exists(indices_path):
        train_subset_idx = np.load(indices_path)
        print('Loaded existing fixed train subset: {} indices from {}'.format(
            len(train_subset_idx), indices_path))
    else:
        rng_idx = np.random.RandomState(TRAIN_SUBSET_SEED)
        train_subset_idx = np.sort(
            rng_idx.choice(np.arange(len(profiles_train)), size=N_TRAIN_SUBSET, replace=False))
        if not os.path.isdir(MODELS_ROOT):
            os.makedirs(MODELS_ROOT)
        np.save(indices_path, train_subset_idx)
        print('Generated and saved NEW fixed train subset: {} indices to {}'.format(
            len(train_subset_idx), indices_path))

    if BIAS:
        scores_train = fairCV['Biased Labels Train (Gender)']
        scores_test = fairCV['Biased Labels Test (Gender)']
    else:
        scores_train = fairCV['Blind Labels Train']
        scores_test = fairCV['Blind Labels Test']

    cand_competencies_train = profiles_train[:, 4:11]
    cand_competencies_test = profiles_test[:, 4:11]

    if 'roberta' in model_name:
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)

    trained_ok = []
    missing_files = []
    failed = []

    for cfg in CONFIGURATIONS:
        tag = '{}_{}'.format(cfg['name'], cfg['entity'])
        print('\n' + '=' * 70)
        print('CONFIGURATION: {}'.format(tag))
        print('=' * 70)

        # --- Check both files exist before doing anything else ---
        missing_here = []
        if not os.path.exists(cfg['train_file']):
            missing_here.append(cfg['train_file'])
        if not os.path.exists(cfg['test_file']):
            missing_here.append(cfg['test_file'])

        if missing_here:
            print('  [SKIPPED] Missing file(s):')
            for f in missing_here:
                print('    - {}'.format(f))
            missing_files.append({'config': tag, 'missing_paths': missing_here})
            continue

        # Initialize all heavy objects to None so the cleanup in `finally`
        # can safely check and delete whatever got created, even if the
        # configuration fails partway through.
        model = optimizer = scheduler = None
        training_loader = testing_loader = None
        train_dataset = test_dataset = None
        raw_anon_train = raw_anon_test = None

        try:
            start = time.time()

            raw_anon_train = np.load(cfg['train_file'], allow_pickle=True)
            raw_anon_test = np.load(cfg['test_file'], allow_pickle=True)

            bios_train = raw_anon_train[:, 1]
            bios_test = raw_anon_test[:, 0]

            assert len(bios_train) == len(profiles_train), (
                f"Mismatch: {len(bios_train)} train bios vs {len(profiles_train)} train profiles."
            )
            assert len(bios_test) == len(profiles_test), (
                f"Mismatch: {len(bios_test)} test bios vs {len(profiles_test)} test profiles."
            )

            # Subsample TRAIN to the fixed set of 4000 candidates computed
            # above (shared by every configuration in this run).
            bios_train_subset = bios_train[train_subset_idx]
            scores_train_subset = scores_train[train_subset_idx]
            comp_train_subset = cand_competencies_train[train_subset_idx]
            names_train_subset = names_train[train_subset_idx]

            print('Using a fixed subset of {} train candidates (out of {})'.format(
                len(train_subset_idx), len(bios_train)))

            # Store data in a dataframe
            train_data = pd.DataFrame(data = {'Text': bios_train_subset, 'Label': scores_train_subset,
                                              'Name': names_train_subset},
                                      columns = ['Text', 'Label','Name'])
            test_data = pd.DataFrame(data = {'Text': bios_test, 'Label': scores_test,
                                             'Name': names_test},
                                      columns = ['Text', 'Label','Name'])

            # Remove punctuation (same as original script -- no gendered
            # word masking here, since the original script didn't apply it)
            train_data.Text = train_data.Text.apply(lambda x: preprocessing(x))
            # Data preprocessing to mask proxy words
            train_data.Text = train_data.apply(lambda x: remove_gendered_words(x.Text, x.Name), axis=1)
            test_data.Text = test_data.Text.apply(lambda x: preprocessing(x))

            print('Dataset de entrenamiento: {}'.format(len(train_data)))
            print('Dataset de test: {}'.format(len(test_data)))

            # ------------------------------------------------------------
            # Save the EXACT preprocessed text used to feed the model,
            # so the MIA attack script can load it directly instead of
            # reconstructing it independently. This removes any risk of
            # a silent preprocessing mismatch between training and attack.
            # ------------------------------------------------------------
            save_dir_text = os.path.join(MODELS_ROOT, model_name, tag)
            if not os.path.isdir(save_dir_text):
                os.makedirs(save_dir_text)
            np.save(os.path.join(save_dir_text, 'train_text_processed.npy'),
                    train_data.Text.values, allow_pickle=True)
            np.save(os.path.join(save_dir_text, 'test_text_processed.npy'),
                    test_data.Text.values, allow_pickle=True)
            print('  Saved exact processed train/test text to: {}'.format(save_dir_text))

            # ------------------------------------------------------------
            # ALSO build and save an anonymized + gender-masked version of
            # TEST (not used for training/validation, only for later MIA
            # Option A analysis, which needs train and test in an
            # identical format). This lets the MIA script be fully
            # self-contained: it only needs to read from MODELS_ROOT, with
            # no dependency on ANON_ROOT or a separate CONFIGURATIONS list.
            # ------------------------------------------------------------
            bios_test_anon = raw_anon_test[:, 1]
            test_data_for_mia = pd.DataFrame(data={'Text': bios_test_anon, 'Name': names_test},
                                              columns=['Text', 'Name'])
            test_data_for_mia.Text = test_data_for_mia.Text.apply(lambda x: preprocessing(x))
            test_data_for_mia.Text = test_data_for_mia.apply(
                lambda x: remove_gendered_words(x.Text, x.Name), axis=1)
            np.save(os.path.join(save_dir_text, 'test_text_for_mia_optionA.npy'),
                    test_data_for_mia.Text.values, allow_pickle=True)
            print('  Saved MIA-ready (anonymized+gender-masked) test text to: {}'.format(save_dir_text))

            # Define custom datasets for the data
            train_dataset = CustomDataset(train_data, comp_train_subset, tokenizer, MAX_LEN)
            test_dataset = CustomDataset(test_data, cand_competencies_test, tokenizer, MAX_LEN)

            # Define dataloader objects over the previous datasets
            train_params = {'batch_size': TRAIN_BATCH_SIZE,
                            'shuffle': True,
                            'num_workers': 0
                            }
            test_params = {'batch_size': TEST_BATCH_SIZE,
                           'shuffle': False,
                           'num_workers': 0
                           }

            training_loader = DataLoader(train_dataset, **train_params)
            testing_loader = DataLoader(test_dataset, **test_params)

            # Init the model and load it in the physical device
            model = BERTRegression(model_name, pool_embeddings = POOLING)
            model.to(device)

            # Define optimizer and scheduler
            optimizer = AdamW(params = model.parameters(), lr = LEARNING_RATE, eps = EPS,
                              weight_decay = 0.005)
            total_steps = len(training_loader) * EPOCHS
            scheduler = get_linear_schedule_with_warmup(optimizer,
                         num_warmup_steps=0, num_training_steps=total_steps)

            # Define the loss functions
            loss = torch.nn.MSELoss()

            # Train the model. NOTE: per-epoch validation over the full
            # test set (4800 samples) was removed here to keep training
            # time comparable to overfit_demo.py, whose training loop
            # also does not validate every epoch. A single validation
            # pass is run once, after training, purely for the printed
            # RMSE (does not affect training itself).
            for epoch in range(EPOCHS):
                train(epoch, model, optimizer, scheduler, training_loader, device, loss)

            print('Running a single post-training validation pass...')
            validate(EPOCHS - 1, model, testing_loader, device, loss)

            # --- Save the model, one subfolder per (anonymizer, entity) ---
            save_dir = os.path.join(MODELS_ROOT, model_name, tag)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            save_file = 'model_unbiased.pt' if not BIAS else 'model_biased.pt'
            if not POOLING:
                save_file = save_file.replace('model', 'model_cls')

            save_path = os.path.join(save_dir, save_file)
            torch.save(model.state_dict(), save_path)
            print('Model saved to: {}'.format(save_path))
            print('Time: {:.1f}s'.format(time.time() - start))

            trained_ok.append({'config': tag, 'model_path': save_path})

        except Exception as e:
            print('  !! FAILED: {}'.format(e))
            traceback.print_exc()
            failed.append({'config': tag, 'error': str(e)})

        finally:
            # ------------------------------------------------------------
            # Full memory cleanup after EVERY configuration, whether it
            # succeeded or failed.
            # ------------------------------------------------------------
            print('  Cleaning up memory before next configuration...')

            if optimizer is not None:
                del optimizer
            if scheduler is not None:
                del scheduler
            if model is not None:
                model.to('cpu')
                del model
            if training_loader is not None:
                del training_loader
            if testing_loader is not None:
                del testing_loader
            if train_dataset is not None:
                del train_dataset
            if test_dataset is not None:
                del test_dataset
            if raw_anon_train is not None:
                del raw_anon_train
            if raw_anon_test is not None:
                del raw_anon_test

            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            print('  Memory cleanup done.')

    # ============================================================
    # Final summary
    # ============================================================
    print('\n' + '=' * 70)
    print('FINAL SUMMARY')
    print('=' * 70)

    print('\n✅ Trained successfully ({}):'.format(len(trained_ok)))
    for item in trained_ok:
        print('  - {} -> {}'.format(item['config'], item['model_path']))

    print('\n⚠️  Skipped (missing files) ({}):'.format(len(missing_files)))
    for item in missing_files:
        print('  - {}:'.format(item['config']))
        for p in item['missing_paths']:
            print('      missing: {}'.format(p))

    if failed:
        print('\n❌ Failed with an error ({}):'.format(len(failed)))
        for item in failed:
            print('  - {}: {}'.format(item['config'], item['error']))