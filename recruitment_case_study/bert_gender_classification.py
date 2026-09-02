# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:23:10 2023

@author: Puesto-1
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics
from transformers import BertTokenizer, BertModel, BertConfig, set_seed, AdamW, get_linear_schedule_with_warmup
from transformers import RobertaModel, RobertaTokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

##Ejemplo: Si estás prediciendo el género y el modelo acierta en el 90% de los casos, tu accuracy será 0.90.

# CUDA flag to allow more detailed error reports 
CUDA_LAUNCH_BLOCKING = 1

# Data preprocessing to remove punctuation
def preprocessing(example):
    example = example.lower()
    return re.sub(r'[^\w\s]','', example)

# Data preprocessing to mask proxy words
def remove_gendered_words(example, name):
    mask_token = '[MASK]'
    gendered = ['he', 'she', 'mr', 'ms', 'her', 'his', 'him','husband','children',
                'family'] + [name.lower()]
    example = example.lower().split(' ')
    example = ' '.join([w if w not in gendered else mask_token for w in example])
    
    return example

# Custom Dataset
class CustomDataset(Dataset):

    # Init routine, we save the datasetm the tokenizer and the context lenght
    def __init__(self, text_data, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = text_data
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
            'targets': torch.tensor(self.data.Label[index], dtype=torch.float)
        }

    
# We define a custom classification model based on the BERT architecture
class BERTMulticlassification(torch.nn.Module):
    
    # Rutina init
    def __init__(self, model_name, hidden_size = 300, drop_rate = 0.3,
                 pool_embedding = True, n_classes = 1):
        
        super(BERTMulticlassification, self).__init__()
        
        # Select the Transformer backbone
        if 'roberta' in model_name:
            self.backbone = RobertaModel.from_pretrained(model_name)
        else:
            self.backbone = BertModel.from_pretrained(model_name)
        
        # Freeze the backbone
        for _, param in self.backbone.named_parameters():
            param.requires_grad = False
        
        # Definition of the classification head
        self.l1 = torch.nn.Linear(self.backbone.config.hidden_size, hidden_size)
        self.d1 = torch.nn.Dropout(drop_rate)
        self.l2 = torch.nn.Linear(hidden_size, 20)
        self.d2 = torch.nn.Dropout(drop_rate)
        self.l3 = torch.nn.Linear(20, n_classes)

        
        # Use the pooling layer as text representation, or the output for the CLS token
        if pool_embedding:
            self.pooling = True
        else:
            self.pooling = False
    
    # Forward routine
    def forward(self, ids, mask = None, token_type_ids = None):
        
        # Extract text embeddings
        sentence_emb = self.backbone(ids, attention_mask = mask, token_type_ids = token_type_ids)
        if self.pooling:
            hidden_1 = self.l1(sentence_emb[1])
        else:
            hidden_1 = self.l1(sentence_emb[0][:,0,:])
           
        hidden_2 = self.d1(torch.sigmoid(hidden_1))
        hidden_3 = self.l2(hidden_2)
        hidden_4 = self.d2(torch.sigmoid(hidden_3))
        output = self.l3(hidden_4)
        
        return output
    
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
        targets = data['targets'].to(device, dtype = torch.float)
        
        # Forward pass
        outputs = model(ids, mask, token_type_ids)
        
        # Set gradients at zero
        optimizer.zero_grad()
        
        # Compute batch loss, backward pass and update the weights
        batch_loss = loss(torch.sigmoid(outputs.squeeze()), targets.squeeze())
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
    
    # We don't store gradient tensors during validation
    with torch.no_grad():
        for _, data in enumerate(test_loader, 0):
            
            # Get batch data
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            
            # Forward pass
            outputs = model(ids, mask, token_type_ids)
            
            # Compute validation loss
            batch_loss = loss(torch.sigmoid(outputs.squeeze()), targets.squeeze())
            val_loss += batch_loss.item() * ids.size(0)
            
            # Store targets and predicted scores
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
            
    # Print validation loss        
    print('Epoch: {}, Val Loss: {:.5f}'.format(epoch, val_loss/len(test_loader.dataset)))
            
    return fin_outputs, fin_targets


if __name__ == '__main__':
    
    data_path =r'\\150.244.56.196\Compartir\NLP\NLPCV\BiasCV\data\ChatGPT\4\PER'
    data_file = 'FairCVdbPER.npy'

    # Parámetros de aprendizaje
    TRAIN_BATCH_SIZE = 32
    TEST_BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 1e-03
    MAX_LEN = 256
    EPS = 1e-08
    POOLING = True
    BERT = True
    W_REMOVAL = True
    
    # Set seed fo reproducibility purposes
    set_seed(123)
    
    # Select physical device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Select backbone
    if BERT:
        model_name = 'bert-base-uncased'
    else:    
        model_name = 'roberta-base'
    i = 0

    full_data_file = os.path.join(data_path, data_file)
    fairCV = np.load(full_data_file, allow_pickle = True).item()
    
    # Train data
    bios_train = fairCV['Bios Train']
    profiles_train = fairCV['Profiles Train']
    names_train = fairCV['Names Train']
    
    # Test data
    bios_test = fairCV['Bios Test']
    profiles_test = fairCV['Profiles Test']
    names_test = fairCV['Names Test']
    
    # Gender labels
    labels_train = profiles_train[:,1]
    labels_test = profiles_test[:,1]
    
    # Select the raw bios (with explicit gender indicators, see De-Arteaga et. al)
    bios_train = bios_train[:,1]
    bios_test = bios_test[:,0]
    
    # Store data in a dataframe
    train_data = pd.DataFrame(data = {'Text': bios_train, 'Label': labels_train,
                                    'Name': names_train},
                            columns = ['Text', 'Label','Name'])
    test_data = pd.DataFrame(data = {'Text': bios_test, 'Label': labels_test,
                                    'Name': names_test},
                            columns = ['Text', 'Label','Name'])
    
    # One-hot and categorical mapping dicts
    label_dict = {0:[1,0], 1: [0,1]}
    label_dict_reverse = {0:'Male', 1:'Female'}
        
    # Remove punctuation
    train_data.Text = train_data.Text.apply(lambda x: preprocessing(x))
    test_data.Text = test_data.Text.apply(lambda x: preprocessing(x))
    
    # Remove proxy words
    if W_REMOVAL:
        train_data.Text = train_data.apply(lambda x: remove_gendered_words(x.Text, x.Name), axis = 1)
        # test_data.Text = test_data.apply(lambda x: remove_gendered_words(x.Text, x.Name), axis = 1)
    
    print('Dataset de entrenamiento: {}'.format(len(train_data)))
    print('Dataset de test: {}'.format(len(test_data)))
    
    # Load tokenier
    if 'roberta' in model_name:
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        
    # Define custom datasets for the data
    train_dataset = CustomDataset(train_data, tokenizer, MAX_LEN)
    test_dataset = CustomDataset(test_data, tokenizer, MAX_LEN)
    
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
    model = BERTMulticlassification(model_name,  pool_embedding = POOLING)
    model.to(device)
    
    # Define optimizer and scheduler
    optimizer = AdamW(params = model.parameters(), lr = LEARNING_RATE, eps = EPS)
    total_steps = len(training_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer,       
                num_warmup_steps=0, num_training_steps=total_steps)
    
    # Define de loss function
    loss = torch.nn.BCELoss()
    
    # Train and validate the model
    for epoch in range(EPOCHS):
        
        # Train one epoch
        train(epoch, model, optimizer, scheduler, training_loader, device, loss)
        
        # Validate
        outputs, targets = validate(epoch, model, testing_loader, device, loss)
        outputs = np.round(outputs)
        accuracy = metrics.accuracy_score(targets, outputs)
        f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
        f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
        cm = metrics.confusion_matrix(targets, outputs)
        print('Accuracy Score = {:.2f}'.format(accuracy))
        print('F1 Score (Micro) = {:.2f}'.format(f1_score_micro))
        print('F1 Score (Macro) = {:.2f}'.format(f1_score_macro))


    # Crear el nombre del archivo con la ruta del directorio
    output_file_name = os.path.join(data_path, f'occupation_results_{i + 1}.txt')   
    # Abrir el archivo para escribir
    with open(output_file_name, 'w') as file:
        # Escribir los resultados en el archivo
        file.write(str(accuracy))
        file.write(str(f1_score_micro))
        file.write(str(f1_score_macro))
    i=i+1           

        # # Visualize prob distribution
        # outputs, _ = validate(epoch, model, testing_loader, device, loss)
        # outputs = np.asarray(outputs).squeeze()
        # df = pd.DataFrame(data = {'Label': labels_test, 'Pred':outputs}, 
        #                 columns = ['Label', 'Pred'])
        # df['Labels_cat'] = df.Label.apply(lambda x: label_dict_reverse[x])
        
        # sns.set()
        # ax = sns.displot(data = df, x = 'Pred', hue = 'Labels_cat', kind = 'kde', palette = 'coolwarm', alpha = .5,
        #                 linewidth = 0, fill = True)
        # sns.move_legend(ax, 'lower center', bbox_to_anchor = (.5,.95), ncol = 4, frameon = False, title = None)
        # #plt.show()
            
