# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 18:44:58 2023

@author: Alejandro

BASELINE training script: trains the model with NO privacy protection
at all -- no NER-based entity anonymization, no gender-word masking.
This is the reference point to compare against: if the MIA detects
significant membership leakage here but NOT on the anonymized models,
that is direct evidence (as requested by Reviewer 2) that anonymization
reduces information leakage.

Saves model + text files using the SAME convention as the other
configurations, under models/<model_name>/Baseline/, so that
miaopA_fixed.py's automatic discovery picks it up with no changes.
"""
import time
import os
import re

import pandas as pd
import numpy as np

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
    return re.sub(r'[^\w\s]', '', example)


# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, text_data, comp, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = text_data
        self.comp = comp
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data.Text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'comp': torch.tensor(self.comp[index, :], dtype=torch.float),
            'targets': torch.tensor(self.data.Label[index], dtype=torch.float)
        }


# BERT-based regression model for the scoring tool
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

    def forward(self, ids, comp, mask=None, token_type_ids=None):
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

        return outputs, hidden_2

    def average_pooling(self, model_output, mask):
        token_emb = model_output[0]
        input_mask_expanded = mask.unsqueeze(-1).expand(token_emb.size()).float()
        sentence_emb = torch.sum(token_emb * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)
        sentence_emb = F.normalize(sentence_emb, p=2, dim=1)
        return sentence_emb


def train(epoch, model, optimizer, scheduler, train_loader, device, loss):
    running_loss = 0.0
    epoch_loss = 0.0

    model.train()
    for ix, data in enumerate(train_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        comp = data['comp'].to(device, dtype=torch.float)
        targets = data['targets'].to(device, dtype=torch.float)

        outputs, _ = model(ids, comp, mask, token_type_ids)

        optimizer.zero_grad()
        batch_loss = torch.sqrt(loss(outputs.squeeze(), targets.squeeze()))
        batch_loss.backward()
        optimizer.step()
        scheduler.step()

        total_batch_loss = batch_loss.item() * ids.size(0)
        running_loss += total_batch_loss
        epoch_loss += total_batch_loss
        if ix % 100 == 0:
            print('Epoch: {}-{}, Loss: {:.5f}'.format(epoch, ix, running_loss / (100 * ids.size(0))))
            running_loss = 0.0

    print('Epoch: {}, Train Loss: {:.5f}'.format(epoch, epoch_loss / len(train_loader.dataset)))


def validate(epoch, model, test_loader, device, loss):
    fin_targets = []
    fin_outputs = []
    val_loss = 0.0

    model.eval()
    with torch.no_grad():
        for _, data in enumerate(test_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            comp = data['comp'].to(device, dtype=torch.float)
            targets = data['targets'].to(device, dtype=torch.float)

            outputs, _ = model(ids, comp, mask, token_type_ids)

            batch_loss = torch.sqrt(loss(outputs.squeeze(), targets.squeeze()))
            val_loss += batch_loss.item() * ids.size(0)

            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())

    print('Epoch: {}, Val Loss: {:.5f}'.format(epoch, val_loss / len(test_loader.dataset)))
    return fin_outputs, fin_targets


if __name__ == '__main__':
    start = time.time()

    # ============================================================
    # Data paths -- BASELINE uses the ORIGINAL FairCVdb.npy directly.
    # No anonymization files needed at all.
    # ============================================================
    ORIGINAL_DATA_FILE = r'D:\Gonzalo\Compartir\NLP\NLPEGNA\code\data\FairCVdb.npy'
    MODELS_ROOT = r'D:\Gonzalo\Compartir\NLP\NLPCV\MIA\data\models'

    # Learning parameters (kept identical to the anonymized configurations
    # for a fair, apples-to-apples comparison in the MIA analysis)
    TRAIN_BATCH_SIZE = 8
    TEST_BATCH_SIZE = 8
    EPOCHS = 20
    LEARNING_RATE = 1e-03
    MAX_LEN = 64
    EPS = 1e-08
    POOLING = True
    BERT = True
    BIAS = True

    set_seed(123)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Using device: {}'.format(device))

    model_name = 'bert-base-uncased' if BERT else 'roberta-base'

    # Load FairCV data
    fairCV = np.load(ORIGINAL_DATA_FILE, allow_pickle=True).item()

    bios_train_full = fairCV['Bios Train']
    profiles_train = fairCV['Profiles Train']
    names_train = fairCV['Names Train']

    bios_test_full = fairCV['Bios Test']
    profiles_test = fairCV['Profiles Test']
    names_test = fairCV['Names Test']

    if BIAS:
        scores_train = fairCV['Biased Labels Train (Gender)']
        scores_test = fairCV['Biased Labels Test (Gender)']
    else:
        scores_train = fairCV['Blind Labels Train']
        scores_test = fairCV['Blind Labels Test']

    cand_competencies_train = profiles_train[:, 4:11]
    cand_competencies_test = profiles_test[:, 4:11]

    # NO anonymization, NO gender-word masking: raw text with explicit
    # gender indicators. Using column 0 for BOTH train and test, since
    # column 1 may already come gender-neutralized from the dataset
    # itself (De-Arteaga et al. convention) -- for a TRUE unprotected
    # baseline we need the most exposed version on both sides.
    bios_train = bios_train_full[:, 0]
    bios_test = bios_test_full[:, 0]

    # Store data in a dataframe
    train_data = pd.DataFrame(data={'Text': bios_train, 'Label': scores_train, 'Name': names_train},
                               columns=['Text', 'Label', 'Name'])
    test_data = pd.DataFrame(data={'Text': bios_test, 'Label': scores_test, 'Name': names_test},
                              columns=['Text', 'Label', 'Name'])

    # Only basic preprocessing -- NO entity anonymization, NO gender-word
    # masking. This is the true, unprotected baseline.
    train_data.Text = train_data.Text.apply(lambda x: preprocessing(x))
    test_data.Text = test_data.Text.apply(lambda x: preprocessing(x))

    print('Dataset de entrenamiento: {}'.format(len(train_data)))
    print('Dataset de test: {}'.format(len(test_data)))

    # ------------------------------------------------------------
    # Save the exact text used, following the SAME convention as the
    # anonymized configurations, so miaopA_fixed.py's auto-discovery
    # picks this up with zero changes.
    # ------------------------------------------------------------
    tag = 'Baseline'
    save_dir = os.path.join(MODELS_ROOT, model_name, tag)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'train_text_processed.npy'),
            train_data.Text.values, allow_pickle=True)
    # For the baseline, the MIA-ready test text is simply the same raw,
    # preprocessed test text -- there is no additional anonymization
    # step to apply here, unlike the anonymized configurations.
    np.save(os.path.join(save_dir, 'test_text_for_mia_optionA.npy'),
            test_data.Text.values, allow_pickle=True)
    print('  Saved train/test text (no anonymization) to: {}'.format(save_dir))

    # Load tokenizer
    if 'roberta' in model_name:
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)

    train_dataset = CustomDataset(train_data, cand_competencies_train, tokenizer, MAX_LEN)
    test_dataset = CustomDataset(test_data, cand_competencies_test, tokenizer, MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
    test_params = {'batch_size': TEST_BATCH_SIZE, 'shuffle': False, 'num_workers': 0}

    training_loader = DataLoader(train_dataset, **train_params)
    testing_loader = DataLoader(test_dataset, **test_params)

    model = BERTRegression(model_name, pool_embeddings=POOLING)
    model.to(device)

    optimizer = AdamW(params=model.parameters(), lr=LEARNING_RATE, eps=EPS, weight_decay=0.005)
    total_steps = len(training_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    loss = torch.nn.MSELoss()

    for epoch in range(EPOCHS):
        train(epoch, model, optimizer, scheduler, training_loader, device, loss)
        validate(epoch, model, testing_loader, device, loss)

    # Save the model
    save_file = 'model_unbiased.pt' if not BIAS else 'model_biased.pt'
    if not POOLING:
        save_file = save_file.replace('model', 'model_cls')

    save_path = os.path.join(save_dir, save_file)
    torch.save(model.state_dict(), save_path)
    print('Model saved to: {}'.format(save_path))
    print('Tiempo: {}'.format(time.time() - start))
