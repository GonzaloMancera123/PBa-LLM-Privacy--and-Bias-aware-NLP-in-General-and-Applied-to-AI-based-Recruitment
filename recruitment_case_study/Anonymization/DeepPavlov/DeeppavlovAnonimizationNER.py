
from deeppavlov import build_model
import numpy as np
import os
from TorchCRF import CRF

# Función para reemplazar entidades que no sean 'LOC' ni 'PER' por '[MASK]'
def replace_non_loc_per_with_mask(text):
    # Ejecutar el modelo NER en el texto
    result = ner_model([text])
    
    # Obtener las entidades etiquetadas
    words = result[0][0]  # Palabras del texto
    tags = result[1][0]   # Etiquetas IOB
    
    # Crear una copia del texto original para reemplazar
    masked_text = text
    
    # Reemplazar las palabras etiquetadas que no sean 'LOC' ni 'PER' por '[MASK]'
    for word, tag in zip(words, tags):
        # if 'LOC' not in tag and 'PER' not in tag and 'O-' not in tag:  # Si la etiqueta no es 'LOC' ni 'PER'
        #     print(f"Etiqueta encontrada: {tag}, palabra: {word}")
        #     masked_text = masked_text.replace(word, '[MASK]', 1)  # Reemplazar solo la primera aparición
    
        # Comentado: Reemplazar entidades 'PER'
        if 'LOC' in tag or 'LOC' in tag:  # Si la etiqueta es de tipo 'PER' (persona)
            masked_text = masked_text.replace(word, '[MASK]', 1)  # Reemplazar solo la primera aparición
    
    return masked_text


# # Ruta donde está almacenado el archivo de la base de datos
data_path = "//150.244.56.196/Compartir/NLP/NLPEGNA/codepegna/data"
database_file = "FairCVdb.npy"

# Cargar el archivo FairCV y extraer los datos de entrenamiento
fairCV = np.load(os.path.join(data_path, database_file), allow_pickle=True).item()
bios_train = fairCV['Bios Test']

# Construir el modelo NER preentrenado de DeepPavlov
ner_model = build_model('ner_conll2003_bert', download=False)

# Array para almacenar los resultados (texto original y anonimizado)
anonymized_bios_train = []
i=0
# Procesar cada biografía en la base de datos
for bio in bios_train:
    original_text = bio[0]  # Texto original de la biografía
    masked_text = replace_non_loc_per_with_mask(original_text)  # Texto anonimizado
    i=i+1
    print(i)
    # Guardar en el nuevo array el texto original en la posición 0 y el anonimizado en la 1
    anonymized_bios_train.append([original_text, masked_text])

# Guardar el nuevo array en un archivo .npy
output_file = "anonymized_fairCVLOC.npy"
np.save(os.path.join(data_path, output_file), anonymized_bios_train)

# Mensaje de confirmación
print(f"Base de datos anonimizada guardada en: {os.path.join(data_path, output_file)}")
