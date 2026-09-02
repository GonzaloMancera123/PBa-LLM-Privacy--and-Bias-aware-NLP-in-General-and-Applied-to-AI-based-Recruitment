import os
import pandas as pd
import stanza
import torch

# Check if PyTorch detects the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# Load the Stanza pipeline to use GPU
stanza.download('en')  # Download the English model
nlp = stanza.Pipeline('en', processors='tokenize,ner', device=device)

def replace_entities_with_mask_stanza(text, nlp, entity_type):
    text = str(text)  # Ensure text is a string
    doc = nlp(text)  # Process text with Stanza
    
    entities_to_replace = []
    for ent in doc.ents:
        if ent.type == entity_type:  # Use 'type' for Stanza
            entities_to_replace.append((ent.start_char, ent.end_char))
    
    for start, end in sorted(entities_to_replace, reverse=True):
        text = text[:start] + '[MASK]' + text[end:]
    
    return text, len(entities_to_replace)

# List of file paths for the CSV files
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
    text_column = df.columns[0]  # Assume the first column contains the text
    
    for entity_type in ["PERSON", "GPE", "ORG"]:  # Standard Stanza labels for people, locations, and organizations
        df_anonymized = df.copy()
        
        # Apply anonymization using Stanza
        df_anonymized[text_column] = df_anonymized[text_column].apply(lambda x: replace_entities_with_mask_stanza(x, nlp, entity_type)[0])
        
        # Create text and entity count columns
        df_anonymized[['text', 'entity_count']] = pd.DataFrame(df_anonymized[text_column].apply(lambda x: replace_entities_with_mask_stanza(x, nlp, entity_type)).tolist(), index=df_anonymized.index)
        total_entities = df_anonymized['entity_count'].sum()
        
        # Get the directory and base filename to construct the new output path
        base_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path).replace('.csv', f'_{entity_type}anonimizado_Stanza.csv')
        
        # Create the full output file path
        new_file_path = os.path.join(base_dir, file_name)
        
        # Save the CSV file with the Stanza anonymization applied
        df_anonymized.drop(columns=['entity_count']).to_csv(new_file_path, index=False)
        
        print(f"File saved: {new_file_path}")
        print(f"Total {entity_type} entities anonymized in {new_file_path}: {total_entities}")

print("Stanza anonymization completed.")