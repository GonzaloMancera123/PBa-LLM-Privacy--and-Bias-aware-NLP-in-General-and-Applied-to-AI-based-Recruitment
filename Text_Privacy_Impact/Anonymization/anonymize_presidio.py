import os
import numpy as np
import pandas as pd
import torch
from presidio_analyzer import AnalyzerEngine

# Configurar para usar la GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")  # Confirmar si usa GPU 0 o CPU

# Configurar el motor de análisis de entidades PII
analyzer = AnalyzerEngine()

def replace_entities_with_mask(text, entity_type):
    text = str(text)  # Asegurar que el texto sea string
    result = analyzer.analyze(text=text, entities=[entity_type], language='en')
    
    entities_to_replace = []
    for entity in result:
        entities_to_replace.append((entity.start, entity.end))
    
    for start, end in sorted(entities_to_replace, reverse=True):
        text = text[:start] + '[MASK]' + text[end:]
    
    return text, len(entities_to_replace)

# Lista de rutas de los archivos CSV
datas_path = [
    r'C:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\BBCNewsSummary\bbc_4_anonimizar.csv'
    # r'C:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\Cyberbullying\cyberbullying_tweets.csv',
    # r'C:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\DbPedia\DBPEDIA_4_anonimizar.csv',
    # r'C:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\IMDB\IMDB_4_anonimizar.csv',
    # r'C:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\NewsCategoryDataset\news_4_anonimizar.csv',
    # r'C:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\TwitterSentimentAnalysis\twitter_4_anonimizar.csv'
]

for file_path in datas_path:
    df = pd.read_csv(file_path)
    text_column = df.columns[0]  # Se asume que la primera columna es la de texto
    summary_column = df.columns[1]  # Se asume que la segunda columna es la del resumen
    
    for entity_type in ["PERSON", "LOCATION"]:
        df_anonymized = df.copy()
        entity_counts = []
        
        # Aplicar la anonimización con Presidio a la columna de texto
        df_anonymized[text_column] = df_anonymized[text_column].apply(lambda x: replace_entities_with_mask(x, entity_type)[0])
        
        # Aplicar la anonimización con Presidio a la columna de resumen
        df_anonymized[summary_column] = df_anonymized[summary_column].apply(lambda x: replace_entities_with_mask(x, entity_type)[0])
        
        # Crear columnas de texto y conteo de entidades
        df_anonymized[['text', 'entity_count']] = pd.DataFrame(df_anonymized[text_column].apply(lambda x: replace_entities_with_mask(x, entity_type)).tolist(), index=df_anonymized.index)
        
        # Total de entidades de tipo 'entity_type' encontradas
        total_entities = df_anonymized['entity_count'].sum()
        
        # Obtener la carpeta y el nombre base del archivo para construir la nueva ruta de salida
        base_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path).replace('.csv', f'_{entity_type[:3]}anonimizado_PRESIDIO.csv')
        
        # Crear la ruta completa del archivo de salida
        new_file_path = os.path.join(base_dir, file_name)
        
        # Guardar el archivo CSV con la anonimización realizada por Presidio
        df_anonymized.drop(columns=['entity_count']).to_csv(new_file_path, index=False)
        
        print(f"Archivo guardado: {new_file_path}")
        print(f"Total de entidades {entity_type} anonimizadas en {new_file_path}: {total_entities}")

print("Anonimización con Presidio completada.")