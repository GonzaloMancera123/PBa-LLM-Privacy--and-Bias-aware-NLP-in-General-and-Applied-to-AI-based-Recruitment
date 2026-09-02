# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 18:44:58 2023

@author: Alejandro
"""
import time
import os
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
                'family','husband']+ [name.lower()]#,'technology','engineering','economics','health','psychology'] + [name.lower()]
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

if __name__ == '__main__':
    start = time.time()
    # Data paths
    data_path = r'D:\Gonzalo\Compartir\NLP\NLPCV\BiasCV\data\ChatGPT\4\LOC'
    data_file = os.path.join(data_path,'FairCVdbLOC.npy')
    
    # Learning parameters
    TRAIN_BATCH_SIZE = 32
    TEST_BATCH_SIZE = 32
    EPOCHS = 50

    LEARNING_RATE = 1e-03
    MAX_LEN = 256
    EPS = 1e-08
    POOLING = True
    BERT = True
    BIAS = True

  
    # Set seed for reproducibility purposes
    set_seed(123)
    
    # Select physical device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Select backbone
    if BERT:
        model_name = 'bert-base-uncased'
    else:    
        model_name = 'roberta-base'
    
    # Load FairCV data
    fairCV = np.load(data_file, allow_pickle = True).item()
    
    # Train data
    bios_train = fairCV['Bios Train']
    profiles_train = fairCV['Profiles Train']
    names_train = fairCV['Names Train']
    
    # Test data
    bios_test = fairCV['Bios Test']
    profiles_test = fairCV['Profiles Test']
    names_test = fairCV['Names Test']
    # Select the target
    if BIAS:
        scores_train = fairCV['Biased Labels Train (Gender)']
        scores_test = fairCV['Biased Labels Test (Gender)']
    else:
        scores_train = fairCV['Blind Labels Train'] 
        scores_test = fairCV['Blind Labels Test']
    
    # Extract competencies from the profiles
    cand_competencies_train = profiles_train[:,4:11]
    cand_competencies_test = profiles_test[:,4:11]
    
    # Extract gender and labor sector
    gender_train = profiles_train[:,1]
    labor_sector_train = profiles_train[:,3]
    gender_test = profiles_test[:,1]
    labor_sector_test = profiles_test[:,3]
    
    # Select the raw bios (with explicit gender indicators, see De-Arteaga et. al)
    bios_train = bios_train[:,1]
    bios_test = bios_test[:,0]

    # Store data in a dataframe
    train_data = pd.DataFrame(data = {'Text': bios_train, 'Label': scores_train,
                                      'Name': names_train},
                              columns = ['Text', 'Label','Name'])
    test_data = pd.DataFrame(data = {'Text': bios_test, 'Label': scores_test,
                                     'Name': names_test},
                              columns = ['Text', 'Label','Name'])
    
    # Remove punctuation
    train_data.Text = train_data.Text.apply(lambda x: preprocessing(x))
    test_data.Text = test_data.Text.apply(lambda x: preprocessing(x))
    

    print('Dataset de entrenamiento: {}'.format(len(train_data)))
    print('Dataset de test: {}'.format(len(test_data)))
    
    # Load tokenier
    if 'roberta' in model_name:
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        
    # Define custom datasets for the data
    train_dataset = CustomDataset(train_data, cand_competencies_train, tokenizer, MAX_LEN)
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
    
    # Train and validate the model
    for epoch in range(EPOCHS):
        
        # Train one epoch
        train(epoch, model, optimizer, scheduler, training_loader, device, loss)
        
        # Validate
        validate(epoch, model, testing_loader, device, loss)

        
    # Save the model
    save_dir = os.path.join(data_path, 'models', model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    save_file = 'model_unbiasedLOC.pt'
    if BIAS:
        save_file = 'model_biasedLOC.pt'
    if not POOLING:
        save_file = save_file.replace('model', 'model_cls')
    
    save_path = os.path.join(save_dir, save_file)
    torch.save(model.state_dict(), save_path)
        
    # Attribute dicts
    labor_sector_dict = {0.25: 'AV', 0.5: 'JU',
                          0.75: 'HC', 1:'ED'}
    gender_dict = {0:'Male', 1:'Female'}
    
    # Extract predictions
    outputs, targets = validate(epoch, model, testing_loader, device, loss)
    outputs = np.asarray(outputs)
    
    # Store the results in a dataset and map the attributes to categories
    df = pd.DataFrame(data = {'Score':outputs.squeeze(), 'Gender':gender_test,
                              'Labor':labor_sector_test}, columns = ['Score', 'Gender', 'Labor'])
    df.Gender = df.Gender.apply(lambda x: gender_dict[x])
    df.Labor = df.Labor.apply(lambda x: labor_sector_dict[x])
    
    # Visualize results by labor sector 
    sns.set()
    ax = sns.displot(data = df, x = 'Score', hue = 'Labor', kind = 'kde', palette = 'Spectral', alpha = .5,
                      linewidth = 0, fill = True)
    sns.move_legend(ax, 'lower center', bbox_to_anchor = (.5,.95), ncol = 4, frameon = False, title = None)
    plt.show()
    
    # Visualize results by gender
    ax = sns.displot(data = df, x = 'Score', hue = 'Gender', kind = 'kde', palette = 'coolwarm', alpha = .5,
                      linewidth = 0, fill = True)
    sns.move_legend(ax, 'lower center', bbox_to_anchor = (.4,.95), ncol = 2, frameon = False, title = None)
    plt.show()
    ax.figure.savefig('score_gender'+ model_name +'.png', dpi = 500, bbox_inches='tight')
    print('Tiempo: {}'.format(time.time()-start))

    
    
    
    
    
    
    
    
    