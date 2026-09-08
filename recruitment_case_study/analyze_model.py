# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 14:22:52 2023

@author: Puesto-1
"""

import os
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn import metrics
from transformers import BertTokenizer, BertModel, set_seed, get_linear_schedule_with_warmup
from transformers import RobertaModel, RobertaTokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.manifold import TSNE


# CUDA flag to allow more detailed error reports 
CUDA_LAUNCH_BLOCKING = 1

# Data preprocessing to remove punctuation
def preprocessing(example):
    return re.sub(r'[^\w\s]','', example)

# Data preprocessing to mask proxy words
def remove_gendered_words(example, name):
    mask_token = '[MASK]'
    gendered = ['he', 'she', 'mr', 'ms', 'her', 'his','him','husband','children',
                'family', 'husband'] + [name]
    example = example.lower().split(' ')
    example = ' '.join([w if w not in gendered else mask_token for w in example])
    
    return example

# Custom Dataset
class CustomDataset(Dataset):

    # Init routine, we save the datasetm the tokenizer and the context lenght
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
    
# Validation routine            
def validate(model, test_loader, device, loss):
    
    fin_targets=[]
    fin_outputs=[]
    embeddings = []
    val_loss = 0.0
    
    # We don't store gradient tensors during validation
    with torch.no_grad():
        for ix, data in enumerate(test_loader, 0):
            print('batch: {}'.format(ix))
            
            # Get batch data
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            comp = data['comp'].to(device, dtype = torch.float)
            targets = data['targets'].to(device, dtype = torch.float)
            
            # Forward pass
            outputs, hidden_outputs = model(ids, comp, mask, token_type_ids)
            
            # Compute validation loss
            batch_loss = torch.sqrt(loss(outputs.squeeze(), targets.squeeze()))
            val_loss += batch_loss.item() * ids.size(0)
            
            # Store targets and predicted scores
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
            
            # Store latent embeddings
            embeddings.extend(hidden_outputs.cpu().detach().numpy().tolist())
            
    # Print validation loss        
    print('Val Loss: {:.5f}'.format(val_loss/len(test_loader.dataset)))
            
    return fin_outputs, fin_targets, embeddings
    
# Compute the KL divergence, which is only defined if Q!=0 where P !=0 for normalized
# distributions
def kl_divergence(p, q):
    return np.sum(np.where(np.logical_and(p != 0,q != 0), p * np.log(p / q), 0))
    
# Compute distributions from the raw scores, and return the KL divergence
def computeKL(x,y):
    
    p = []
    q = []
    
    # Compute scores distributions
    for i in range(0,98,2):
        i = i/100
        j = i + 0.02
        p.append(np.sum(np.where(np.logical_and(x >= i, x <j), 1, 0)))
        q.append(np.sum(np.where(np.logical_and(y >= i, y <j), 1, 0)))
    p = np.asarray(p).reshape(1,-1)
    q = np.asarray(q).reshape(1,-1)
    
    # Normalize distributions
    p = p/x.shape[0]
    q = q/y.shape[0]
    
    # Return the KL divergence
    return kl_divergence(p,q)

if __name__ == '__main__':
    
    # Data paths 
    data_path = r'\\150.244.56.196\Compartir\NLP\NLPEGNA\code\data'
    data_file = os.path.join(data_path,'FairCVdb.npy')
    
    # Parameters
    TEST_BATCH_SIZE = 32
    MAX_LEN = 256
    N_RANK = 500
    POOLING = True
    BERT = True
    
    BIAS = True
    
    # if APPROACH in ['word','lntl']:
    #     BIAS = True
        
    # Set config info for loading/storing purposes
    config = '_unbiased'
    if BIAS:
        config = '_biased'
        # if APPROACH == 'word':
        #     config = '_biased_words'
        # elif APPROACH == 'lntl':
        #     config = '_biased_lntl'
    if not POOLING:
        config = '_cls' + config
    
    # Set seed fo reproducibility purposes
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
    biased_scores_test = fairCV['Biased Labels Test (Gender)']
    unbiased_scores_test = fairCV['Blind Labels Test']
     
    # Extract competencies from the profiles
    cand_competencies_test = profiles_test[:,4:11]
    
    # Extract gender and labor sector
    gender_test = profiles_test[:,1]
    labor_sector_test = profiles_test[:,3]
    
    # Select the raw bios (with explicit gender indicators, see De-Arteaga et. al)
    bios_test = bios_test[:,0]
    
    # Store data in a dataframe
    test_data = pd.DataFrame(data = {'Text': bios_test, 'Label': biased_scores_test,
                                     'Name': names_test},
                              columns = ['Text', 'Label','Name'])
    
    # Remove punctuation
    test_data.Text = test_data.Text.apply(lambda x: preprocessing(x))
    
    # # Remove proxy words
    # if APPROACH == 'word':
    #     test_data.Text = test_data.apply(lambda x: remove_gendered_words(x.Text, x.Name), axis = 1)

    # Load tokenier
    if 'roberta' in model_name:
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        
    # Define custom datasets for the data
    test_dataset = CustomDataset(test_data, cand_competencies_test, tokenizer, MAX_LEN)
    
    # Define dataloader objects over the previous datasets
    test_params = {'batch_size': TEST_BATCH_SIZE,
                   'shuffle': False,
                   'num_workers': 0
                   }
    
    testing_loader = DataLoader(test_dataset, **test_params)
    
    # Path to the model
    model_dir = r'\\150.244.56.196\Compartir\NLP\NLPCV\BiasCV\data\Presidio\models\bert\PER'
    model_file = 'model_biased.pt'
    
    # Init the model, load the weight and send into the physical device
    model = BERTRegression(model_name, pool_embeddings = POOLING)
    model.load_state_dict(torch.load(os.path.join(model_dir, model_file)))
    model.to(device)
    model.eval()
    
    # Define the loss functions
    loss = torch.nn.MSELoss()
    
    # Extract model predictions
    print('Start validation')
    outputs, targets, embeddings = validate(model, testing_loader, device, loss)
    outputs = np.asarray(outputs)
    embeddings = np.asarray(embeddings)
    
    # Diccionarios de etiquetas
    labor_sector_dict = {0.25: 'AV', 0.5: 'JU',
                          0.75: 'HC', 1:'ED'}
    gender_dict = {0:'Male', 1:'Female'}

    # Store data in a dataframe for visualization purposes
    df = pd.DataFrame(data = {'Score':outputs.squeeze(), 'Gender':gender_test,
                              'Labor':labor_sector_test, 'Label': unbiased_scores_test,
                              'Biased_Label':biased_scores_test},
                      columns = ['Score', 'Gender', 'Labor', 'Label', 'Biased_Label'])
    df.Gender = df.Gender.apply(lambda x: gender_dict[x])
    df.Labor = df.Labor.apply(lambda x: labor_sector_dict[x])
    
    # Save dir
    save_dir = os.path.join(data_path, model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    os.chdir(save_dir)

    # Visualize results by labor sector
    sns.set()
    ax = sns.displot(data = df, x = 'Score', hue = 'Labor', kind = 'kde', palette = 'Spectral', alpha = .5,
                     linewidth = 0, fill = True)
    sns.move_legend(ax, 'lower center', bbox_to_anchor = (.5,.94), ncol = 4, frameon = False, title = None)
    #plt.show()
    ax.figure.savefig('score_occupation'+ config +'.png', dpi = 500, bbox_inches='tight')
    
    # Visualize resutls by gender
    ax = sns.displot(data = df, x = 'Score', hue = 'Gender', kind = 'kde', palette = 'coolwarm', alpha = .5,
                      linewidth = 0, fill = True)
    sns.move_legend(ax, 'lower center', bbox_to_anchor = (.4,.94), ncol = 2, frameon = False, title = None)
    ax.set(xlim=(0,1))
    # ax.set(yticks=[0,0.5,1.0,1.5,2.0])
    #plt.show()
    ax.figure.savefig('score_gender'+ config +'.png', dpi = 500, bbox_inches='tight')
    
    # Compute the KL divergence
    KL = computeKL(df.Score[df.Gender == 'Male'].to_numpy(), df.Score[df.Gender == 'Female'].to_numpy())
    print('KL divergence: {:.4f}'.format(KL))
    
    # Extract the top-k scores, compute gender proportions, and the utility
    df = df.sort_values(by = 'Label', ascending = False)
    df['Top'] = [1] * N_RANK + [0] * (len(df) - N_RANK)
    N_RANK_MALE = df.query("Gender == 'Male'").Top.sum()
    N_RANK_FEMALE = df.query("Gender == 'Female'").Top.sum()
    
    df = df.sort_values(by = 'Score', ascending = False)
    top = df.iloc[:N_RANK]
    
    male_prop = len(top.query("Gender == 'Male'"))
    female_prop = N_RANK - male_prop
    print('Proporción de hombres en el top-{}: {:.2f}%'.format(N_RANK, male_prop/N_RANK*100))
    print('Proporción de mujeres en el top-{}: {:.2f}%'.format(N_RANK, female_prop/N_RANK*100))
    print('Demographic ratio en el top-{} : {:.3f}'.format(N_RANK, min(female_prop,male_prop)/max(female_prop,male_prop)))
    
    recall = top.Top.sum()/N_RANK
    recall_male = top.query("Gender == 'Male'").Top.sum()/N_RANK_MALE
    recall_female = top.query("Gender == 'Female'").Top.sum()/N_RANK_FEMALE
    print('Recall : {:.2f}%'.format(recall * 100))
    print('Recall male : {:.2f}%'.format(recall_male * 100))
    print('Recall female: {:.2f}%'.format(recall_female * 100))
    
    
    # Extract average error by gender
    df['Dif'] = df.Score - df.Biased_Label
    df['Dif_2'] = df.Score - df.Label

    df_male = df.query("Gender == 'Male'")
    df_female = df.query("Gender == 'Female'")
    
    print('Error medio en los scores (sesgados) masculinos: {}'.format(np.mean(df_male.Dif)))
    print('Error medio en los scores (sesgados) femeninos: {}'.format(np.mean(df_female.Dif)))
    print('Error medio en los scores (no sesgados) masculinos: {}'.format(np.mean(df_male.Dif_2)))
    print('Error medio en los scores (no sesgados) femeninos: {}'.format(np.mean(df_female.Dif_2)))
    
    #t-SNE representations with gender annotations 
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 20, n_iter = 500)
    tsne_trans = tsne.fit_transform(embeddings)
    
    tsne_df = pd.DataFrame(data = {'t-SNE-1':tsne_trans[:,0], 't-SNE-2':tsne_trans[:,1],
                                    'Gender':gender_test}, columns = ['t-SNE-1','t-SNE-2','Gender'])
    tsne_df_occupation = pd.DataFrame(data = {'t-SNE-1':tsne_trans[:,0], 't-SNE-2':tsne_trans[:,1],
                                    'Labor':labor_sector_test}, columns = ['t-SNE-1','t-SNE-2','Labor'])
    tsne_df.Gender = tsne_df.Gender.apply(lambda x: gender_dict[x])
    tsne_df_occupation.Labor = tsne_df_occupation.Labor.apply(lambda x: labor_sector_dict[x])
    
    sns.set()
    ax = sns.scatterplot(data = tsne_df, x = 't-SNE-1', y = 't-SNE-2', hue = 'Gender',palette = 'coolwarm')
    sns.move_legend(ax, 'lower center', bbox_to_anchor = (.4,.96), ncol = 2, frameon = False, title = None)
    #plt.show()
    ax.figure.savefig('tsne'+ config +'.png', dpi = 500, bbox_inches='tight')
    
    sns.set()
    ax = sns.scatterplot(data = tsne_df_occupation, x = 't-SNE-1', y = 't-SNE-2', hue = 'Labor',palette = 'Spectral')
    sns.move_legend(ax, 'lower center', bbox_to_anchor = (.4,.96), ncol = 4, frameon = False, title = None)
    #plt.show()
    ax.figure.savefig('tsne_occuparion'+ config +'.png', dpi = 500, bbox_inches='tight')
    
    
    
    
    
    