"""
Created on Tue Jul  4 14:24:44 2023

@author: Puesto-1
"""

import os
import matplotlib
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics
from transformers import BertTokenizer, BertModel, BertConfig, set_seed, AdamW, get_linear_schedule_with_warmup
from transformers import RobertaModel, RobertaTokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Flag de CUDA que permite un report más completo en caso de error 
CUDA_LAUNCH_BLOCKING = 1

# Custom dataset para cargar los datos, heredamos de Dataset
class CustomDataset(Dataset):

    # init, guardamos el dataset, tokenizer y el tamaño de entrada del modelo
    def __init__(self, text_data, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = text_data
        self.max_len = max_len
    
    # Longitud del dataset
    def __len__(self):
        return len(self.data)

    # Función para devolver batches
    def __getitem__(self, index):
        text = str(self.data.Text[index])
        text = " ".join(text.split())

        # Tokenizamos el texto
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding = 'max_length',
            return_token_type_ids=True,
            truncation = True
        )
        
        # Extraemos tokens, máscara y tipo de tokens
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        # Devolvemos diccionario con el resultado del tokenizer, las etiquetas y las 
        # competencias
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.data.Label[index], dtype=torch.float)
        }
    
# Definimos una clase para multiclass con BERT que herede del módulo básico de torch
class BERTMulticlassification(torch.nn.Module):
    
    # Rutina init
    def __init__(self, model_name, hidden_size = 70, n_classes = 4):
        super(BERTMulticlassification, self).__init__()
        
        # Usamos de backbone BERT
        if 'roberta' in model_name:
            self.backbone = RobertaModel.from_pretrained(model_name)
        else:
            self.backbone = BertModel.from_pretrained(model_name)
        
        # Congelamos sus parámetros
        for _, param in self.backbone.named_parameters():
            param.requires_grad = False
        
        # Añadimos un dropout y una capa de clasificación con la entrada de los embeddings
        # de BERT
        self.l1 = torch.nn.Linear(self.backbone.config.hidden_size, hidden_size)
        self.d1 = torch.nn.Dropout(0.5)
        self.l2 = torch.nn.Linear(300, 70)
        self.d2 = torch.nn.Dropout(0.5)
        self.l3 = torch.nn.Linear(70, n_classes)
    
    # Custom forward
    def forward(self, ids, mask, token_type_ids):
        
        # Nos quedamos la segunda salida, que es la del [CLS]
        sentence_emb = self.backbone(ids, attention_mask = mask, token_type_ids = token_type_ids)
        hidden_1 = self.l1(sentence_emb[1])
        hidden_2 = self.d1(F.relu(hidden_1))
        hidden_3 = self.l2(hidden_2)
        hidden_4 = self.d2(F.relu(hidden_3))
        output = self.l3(hidden_4)
        
        return output
    
# Rutina básica de entrenamiento 
def train(epoch, model, optimizer, scheduler, train_loader, device, loss):
    
    model.train()
    running_loss = 0.0
    for ix, data in enumerate(train_loader, 0):
        
        # Obtenemos los datos del batch
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)
        
        # Forward
        outputs = model(ids, mask, token_type_ids)
        
        # ponemos gradientes a cero 
        optimizer.zero_grad()
        
        # calculamos pérdidas, backward y actualizamos pesos
        loss_batch = loss(outputs.squeeze(), targets.squeeze())
        loss_batch.backward()
        optimizer.step()
        scheduler.step()
        
        # Guardamos pérdidas y pintamos estadísticas si toca
        running_loss += loss_batch.item()
        if ix % 100 == 0:
            print('Epoch: {}-{}, Loss: {}'.format(epoch, ix, running_loss))
            running_loss = 0.0
            
def validate(model, test_loader, device):
    
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    
    # Ajustamos torch para no calcular gradientes
    with torch.no_grad():
        for _, data in enumerate(test_loader, 0):
            
            # Obtenemos los datos del batch
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            
            # Forward
            outputs = model(ids, mask, token_type_ids)
            
            # Extraemos scores y targets
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
            
    return fin_outputs, fin_targets


if __name__ == '__main__':
    
    # Lista de directorios
    data_paths = [

        r'\\150.244.56.196\Compartir\NLP\NLPCV\BiasCV\data\ChatGPT\4\LOC',
        r'\\150.244.56.196\Compartir\NLP\NLPCV\BiasCV\data\ChatGPT\4\PER5'
    ]
    # Archivos de datos correspondientes
    data_files = [
        'FairCVdbLOC.npy',
        'FairCVdbPER.npy'
    ]


    # Parámetros de aprendizaje
    TRAIN_BATCH_SIZE = 32
    TEST_BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-03
    MAX_LEN = 256
    EPS = 1e-08
    
    # Ajustamos una semilla fija por reproducibilidad
    set_seed(123)
    
        # Seleccionamos el dispositivo físico de ejecución
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Modelo BERT
    model_name = 'roberta-base'

    # Iteramos sobre cada par de directorio y archivo de datos
    for data_path, data_file in zip(data_paths, data_files):
        full_data_file = os.path.join(data_path, data_file)
        
        # Cargamos datos de FairCV
        fairCV = np.load(full_data_file, allow_pickle=True).item()
        bios_train = fairCV['Bios Train']
        profiles_train = fairCV['Profiles Train']
        bios_test = fairCV['Bios Test']
        profiles_test = fairCV['Profiles Test']
        
        # Etiquetas de sector
        labels_train = profiles_train[:, 3]
        labels_test = profiles_test[:, 3]
        
        # Nos quedamos con las bios normales
        bios_train = bios_train[:, 1]
        bios_test = bios_test[:, 0]
        
        # Pasamos a formato dataframe
        train_data = pd.DataFrame(data={'Text': bios_train, 'Label': labels_train},
                                columns=['Text', 'Label'])
        test_data = pd.DataFrame(data={'Text': bios_test, 'Label': labels_test},
                                columns=['Text', 'Label'])
        
        label_dict = {0.25: [1, 0, 0, 0], 0.5: [0, 1, 0, 0], 0.75: [0, 0, 1, 0], 1: [0, 0, 0, 1]}
        label_dict_reverse = {0.25: 0, 0.5: 1, 0.75: 2, 1: 3}
        
        # label encoding
        train_data['Label'] = train_data.Label.apply(lambda x: label_dict[x])
        test_data['Label_2'] = test_data.Label.apply(lambda x: label_dict_reverse[x])
        test_data['Label'] = test_data.Label.apply(lambda x: label_dict[x])
        
        print('Dataset de entrenamiento: {}'.format(len(train_data)))
        print('Dataset de test: {}'.format(len(test_data)))
        
        # Cargamos el tokenizer
        if 'roberta' in model_name:
            tokenizer = RobertaTokenizer.from_pretrained(model_name)
        else:
            tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Cargamos datasets
        train_dataset = CustomDataset(train_data, tokenizer, MAX_LEN)
        test_dataset = CustomDataset(test_data, tokenizer, MAX_LEN)
        
        # Definimos data loaders sobre estos datasets
        train_params = {'batch_size': TRAIN_BATCH_SIZE,
                        'shuffle': True,
                        'num_workers': 0}

        test_params = {'batch_size': TEST_BATCH_SIZE,
                    'shuffle': False,
                    'num_workers': 0}
        
        training_loader = DataLoader(train_dataset, **train_params)
        testing_loader = DataLoader(test_dataset, **test_params)
        
        # Cargamos el modelo y lo mandamos al dispositivo físico
        model = BERTMulticlassification(model_name, hidden_size=300)
        model.to(device)
        
        # Definimos el optimizador
        optimizer = AdamW(params=model.parameters(), lr=LEARNING_RATE, eps=EPS)
        
        # Definimos el scheduler
        total_steps = len(training_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        
        # Definimos las pérdidas
        loss = torch.nn.CrossEntropyLoss()
        
        # Entrenamos y validamos
        accuracies = []
        best_accuracy = 0.0
        best_model_path = os.path.join(data_path, 'best_model.pth')
        # Inicializamos las variables para F1 Scores
        best_f1_micro = 0.0
        best_f1_macro = 0.0
        for epoch in range(EPOCHS):
            # Entrenamos una época
            train(epoch, model, optimizer, scheduler, training_loader, device, loss)
            
            # Evaluamos
            outputs, targets = validate(model, testing_loader, device)
            outputs = np.argmax(outputs, axis=1)
            targets = np.argmax(targets, axis=1)
            accuracy = metrics.accuracy_score(targets, outputs)
            accuracies.append(accuracy)  # Guardamos el accuracy
            f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
            f1_score_macro = metrics.f1_score(targets, outputs, average='macro')

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_f1_micro = f1_score_micro
                best_f1_macro = f1_score_macro
                torch.save(model.state_dict(), best_model_path)
                print('Nuevo mejor modelo guardado con accuracy: {:.2f}'.format(best_accuracy))
            
            cm = metrics.confusion_matrix(targets, outputs)
            print('Accuracy Score = {:.2f}'.format(accuracy))
            print('F1 Score (Micro) = {:.2f}'.format(f1_score_micro))
            print('F1 Score (Macro) = {:.2f}'.format(f1_score_macro))
            
        # Guardamos el modelo
        # save_path = os.path.join(data_path, 'model_unbiased')
        # torch.save(model.state_dict(), save_path)
        
        # Extraemos género
        gender_train = profiles_train[:, 1]
        gender_test = profiles_test[:, 1]
        gender_dict = {0: 'Male', 1: 'Female'}
        
        # Guardar las precisiones con un número de índice único
        accuracy_file_name = f'accuracies_{data_files.index(data_file) + 1}.npy'
        np.save(os.path.join(data_path, accuracy_file_name), np.array(accuracies))

        # Guardar los F1 Scores con un número de índice único
        f1_scores_file_name = f'f1_scores_{data_files.index(data_file) + 1}.npy'
        np.save(os.path.join(data_path, f1_scores_file_name), {'f1_micro': best_f1_micro, 'f1_macro': best_f1_macro})

        
        # Al final del entrenamiento, carga el mejor modelo para la validación
        model.load_state_dict(torch.load(best_model_path))

        # Extraemos predicciones del modelo
        outputs, targets = validate(model, testing_loader, device)
        outputs = np.argmax(outputs, axis=1)
        targets = np.argmax(targets, axis=1)
        cm = metrics.confusion_matrix(targets, outputs)
        acc_per_class = cm.diagonal() / cm.sum(axis=1)
        
        class_outputs = []

        for ix, labor_sector in enumerate([0.25, 0.5, 0.75, 1]):
            # Sacamos el accuracy de la clase
            class_acc = acc_per_class[ix]

            # Sacamos los df de la clase 
            train_data_class = train_data[labels_train == labor_sector]
            test_data_class = test_data[labels_test == labor_sector]

            sector_prop_train = len(train_data_class) / len(train_data)
            sector_prop_test = len(test_data_class) / len(test_data)

            # Crear el nombre del archivo con la ruta del directorio
            output_file_name = os.path.join(data_path, f'occupation_results_{ix + 1}.txt')
            
            # Abrir el archivo para escribir
            with open(output_file_name, 'w') as file:
                # Escribir los resultados en el archivo
                file.write(f'Occupation: {labor_sector}\n')
                file.write(f'Proportion in train: {sector_prop_train:.2f}\n')
                file.write(f'Proportion in test: {sector_prop_test:.2f}\n')
                file.write(f'Accuracy Score = {class_acc:.2f}\n')
            
            # Guardar los outputs de la clase
            class_outputs.append(outputs[labels_test == labor_sector])



    