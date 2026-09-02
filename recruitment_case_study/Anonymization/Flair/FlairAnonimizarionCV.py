import numpy as np
import os
from flair.data import Sentence
from flair.models import SequenceTagger
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Función para reemplazar entidades 'LOC' y 'PER' por '[MASK]' utilizando Flair
def replace_loc_per_with_mask_flair(text, ner_model):
    # Convertir el texto a string estándar de Python (por si es numpy.str_)
    text = str(text)
    
    # Crear una oración de Flair
    sentence = Sentence(text)
    
    # Ejecutar el modelo NER en el texto
    ner_model.predict(sentence)
    
    # Crear una copia del texto original para reemplazar
    masked_text = text
    
    # Lista para almacenar las entidades con sus posiciones
    entities_to_replace = []

    # Reemplazar las palabras etiquetadas como 'LOC' o 'PER' por '[MASK]'
    for entity in sentence.get_spans('ner'):
        if entity.get_label("ner").value in ['PER']:  # Si es LOC o PER
            # Guardar las posiciones de las entidades a reemplazar
            entities_to_replace.append((entity.start_position, entity.end_position))
    
    # Realizar el reemplazo de manera inversa para no afectar las posiciones
    for start, end in sorted(entities_to_replace, reverse=True):
        masked_text = masked_text[:start] + '[MASK]' + masked_text[end:]
    
    return masked_text

# Ruta donde está almacenado el archivo de la base de datos
data_path = "//150.244.56.196/Compartir/NLP/NLPEGNA/codepegna/data"
database_file = "FairCVdb.npy"

# Cargar el archivo FairCV y extraer los datos de entrenamiento
fairCV = np.load(os.path.join(data_path, database_file), allow_pickle=True).item()
bios_train = fairCV['Bios Test']

# Cargar el modelo NER de Flair preentrenado (inglés)
ner_model = SequenceTagger.load('ner')

# Array para almacenar los resultados (texto original y anonimizado)
anonymized_bios_train = []

i=0
# Procesar cada biografía en la base de datos
for bio in bios_train:
    original_text = bio[0]  # Texto original de la biografía
    masked_text = replace_loc_per_with_mask_flair(original_text, ner_model)  # Texto anonimizado
    print(i)
    # Guardar en el nuevo array el texto original en la posición 0 y el anonimizado en la 1
    anonymized_bios_train.append([original_text, masked_text])
    i = i+1
# Guardar el nuevo array en un archivo .npy
output_file = "anonymized_fairCVPER.npy"

data_save = "//150.244.56.196/Compartir/NLP/NLPCV/Anonimization/Flair/Test"
np.save(os.path.join(data_save, output_file), anonymized_bios_train)

# Mensaje de confirmación
print(f"Base de datos anonimizada guardada en: {os.path.join(data_save, output_file)}")
