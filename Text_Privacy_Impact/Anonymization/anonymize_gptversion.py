import os
import pandas as pd
import openai
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Limit TensorFlow to the second GPU
        tf.config.set_visible_devices(gpus[1], 'GPU')
        print("Using only GPU:", gpus[1])
    except RuntimeError as e:
        print(e)
        
# OpenAI API key configuration
api_key = ""
openai.api_key = api_key

# List of file paths for the CSV files
datas_path = [
    # r'C:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\BBCNewsSummary\bbc_4_anonimizar.csv'
    # r'c:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\Cyberbullying\cyber_4_anonimizar.csv',
    # r'C:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\DbPedia\DBPEDIA_4_anonimizar.csv',
    # r'C:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\IMDB\IMDB_4_anonimizar.csv',
    # r'C:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\NewsCategoryDataset\news_4_anonimizar.csv'
    r'C:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\TwitterSentimentAnalysis\twitter_4_anonimizar.csv'
]

def replace_entities_with_mask(text, max_retries=3):
    input_text = f"""Act as an organization anonymizer. Replace only proper names of organizations or companies (e.g., Apple, Microsoft, Mercadona) in the sentence with [MASK], without modifying names of people, locations, or any other parts of the sentence.

⚠️ Do NOT modify:

Names of people (e.g., John, Sarah)

Names of cities, towns, or countries (e.g., Madrid, Tokyo, Spain)

Articles, prepositions, or any structural elements of the sentence

✅ Only replace names of organizations or companies (e.g., Apple, Microsoft, Mercadona) with [MASK].

📏 IMPORTANT: The output must have exactly the same number of characters as the input sentence. Only replace names of organizations or companies with [MASK], and do not add or remove any other characters. The rest of the sentence must remain unchanged, including spaces, punctuation, and formatting.

🛑 No words should be added or removed, and only the names of organizations or companies should be replaced with [MASK]. The structure of the sentence should stay identical, apart from the changes made by the replacement.

Provide only the modified sentence, without any extra comments or quotation marks.

Examples:

"John works at Apple in the marketing department" -> "John works at [MASK] in the marketing department"

"Microsoft is launching a new product next month" -> "[MASK] is launching a new product next month"

"The project is a collaboration between NASA and Mercadona" -> "The project is a collaboration between [MASK] and [MASK]"

Your turn: {text}
    """
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4.1-mini-2025-04-14",
                messages=[{"role": "user", "content": input_text}],
                max_tokens=len(text.split()) + 5,
                temperature=0.0,
            )
            result = response.choices[0].message['content'].strip()
            return result
        except Exception as e:
            print(f"⚠️ Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))  # Progressive backoff
            else:
                print("❌ All attempts exhausted. Using original text.")
                return text

# Process all files
for file_path in datas_path:
    print(f"\n📂 Processing: {file_path}")
    df_original = pd.read_csv(file_path)
    
    # Determine the reduction percentage based on the file
    if "news_4_anonimizar.csv" in file_path.lower():
        reduction_ratio = 0.90
    elif "DBPEDIA_4_anonimizar.csv" in file_path.lower():
        reduction_ratio = 0.95
    elif "cyber_4_anonimizar.csv" in file_path.lower():
        reduction_ratio = 0.90
    else:
        reduction_ratio = 0.90  # Default value if no file matches
    
    df_original_reduced, _ = train_test_split(
        df_original,
        test_size=(reduction_ratio),
        # stratify=df_original['etiquetas'],  # Keep class proportion
        random_state=42
    )
    df_anonymized = df_original_reduced.copy()

    for index in range(len(df_anonymized)):
        original_text = df_anonymized.iloc[index, 0]
        if pd.isna(original_text):  # Validate if the text is not null
            print(f"Row {index}: null text, skipping.")
            continue
        anonymized_text = replace_entities_with_mask(original_text)
        df_anonymized.iloc[index, 0] = anonymized_text

        # Show progress every row
        if index % 1 == 0:
            print(f"🔄 Progress: {index + 1}/{len(df_original_reduced)} rows processed")

        # Wait slightly between calls to avoid rate limits
        time.sleep(0.14)

    # Save the new CSV
    new_file_path = file_path.replace(".csv", "_ORGanonimizado_CHATGPT4nano.csv")
    df_anonymized.to_csv(new_file_path, index=False)
    print(f"✅ Saved: {new_file_path}")

print("\n🎉 Anonymization completed!")