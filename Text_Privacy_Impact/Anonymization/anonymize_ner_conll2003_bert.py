import os
import numpy as np
import pandas as pd
from deeppavlov import build_model
# Check if CUDA is available
import torch
# Use GPU 1 if available
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Load the DeepPavlov NER model
ner_model = build_model('ner_conll2003_bert', download=True, install=True)

# Function to split text into fragments of suitable length
def split_text(text, max_length=512):
    words = text.split()
    fragments = []
    
    current_fragment = []
    current_length = 0
    
    for word in words:
        current_length += len(word) + 1  # Add 1 to count the space
        if current_length <= max_length:
            current_fragment.append(word)
        else:
            fragments.append(' '.join(current_fragment))
            current_fragment = [word]
            current_length = len(word) + 1  # Reset length with the new word
    
    if current_fragment:
        fragments.append(' '.join(current_fragment))  # Add the last fragment
    
    return fragments

# Function to replace entities with [MASK]
def replace_entities_with_mask(text, entity_type):
    text = str(text)  # Ensure text is a string
    fragments = split_text(text)  # Split text into fragments
    masked_text = ''
    total_entities = 0
    
    # Process each fragment
    for fragment in fragments:
        result = ner_model([fragment])
        
        words = result[0][0]  # Words of the fragment
        tags = result[1][0]   # IOB tags
        
        entities_to_replace = []
        
        # Identify entities that need to be replaced
        for word, tag in zip(words, tags):
            if entity_type in tag:  # If the entity matches the searched type
                start = fragment.find(word)
                end = start + len(word)
                entities_to_replace.append((start, end))
        
        # Replace entities in the fragment
        for start, end in sorted(entities_to_replace, reverse=True):
            fragment = fragment[:start] + '[MASK]' + fragment[end:]
        
        # Count the number of entities replaced in this fragment
        total_entities += len(entities_to_replace)
        
        # Concatenate the processed fragments
        masked_text += fragment + ' '
    
    return masked_text.strip(), total_entities

# List of file paths for the CSV files
datas_path = [
    r'C:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\BBCNewsSummary\bbc_4_anonimizar.csv',
    r'C:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\Cyberbullying\cyberbullying_tweets.csv',
    r'C:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\DbPedia\DBPEDIA_4_anonimizar.csv',
    r'C:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\IMDB\IMDB_4_anonimizar.csv',
    r'C:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\NewsCategoryDataset\news_4_anonimizar.csv',
    r'C:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\TwitterSentimentAnalysis\twitter_4_anonimizar.csv'
]

# Process each CSV file
for file_path in datas_path:
    df = pd.read_csv(file_path)
    text_column = df.columns[0]  # Assume the first column contains the text
    
    for entity_type in ["PER", "LOC"]:
        df_anonymized = df.copy()
        
        # Apply anonymization using DeepPavlov
        df_anonymized[text_column] = df_anonymized[text_column].apply(lambda x: replace_entities_with_mask(x, entity_type)[0])
        
        # Create text and entity count columns
        df_anonymized[['text', 'entity_count']] = pd.DataFrame(df_anonymized[text_column].apply(lambda x: replace_entities_with_mask(x, entity_type)[1]).tolist(), index=df_anonymized.index)
        total_entities = df_anonymized['entity_count'].sum()
        
        # Get the directory and base filename to construct the new output path
        base_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path).replace('.csv', f'_{entity_type}anonimizado_DEEPPAVLOV.csv')
        
        # Create the full path of the output file
        new_file_path = os.path.join(base_dir, file_name)
        
        # Save the CSV file with the anonymization performed by DeepPavlov
        df_anonymized.drop(columns=['entity_count']).to_csv(new_file_path, index=False)
        
        print(f"File saved: {new_file_path}")
        print(f"Total {entity_type} entities anonymized in {new_file_path}: {total_entities}")

print("DeepPavlov anonymization completed.")