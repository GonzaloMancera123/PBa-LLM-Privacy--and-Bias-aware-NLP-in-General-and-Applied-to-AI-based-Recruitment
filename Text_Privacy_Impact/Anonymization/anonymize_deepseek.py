import os
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from openai import OpenAI  # Using the same SDK compatible with DeepSeek

# DeepSeek client configuration
client = OpenAI(
    api_key="",  # Replace with your actual DeepSeek API key
    base_url="https://api.deepseek.com/v1"  # DeepSeek endpoint
)

# List of file paths for the CSV files
datas_path = [
    r'c:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\Cyberbullying\cyber_4_anonimizar.csv',
    r'C:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\DbPedia\DBPEDIA_4_anonimizar.csv',
    r'C:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\IMDB\IMDB_4_anonimizar.csv',
    r'C:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\NewsCategoryDataset\news_4_anonimizar.csv'
]

def replace_entities_with_mask(text, max_retries=3):
    input_text = f"""Act as a person anonymizer. Replace only proper names of people (e.g., John, Sarah, Michael) in the sentence with [MASK], without modifying names of organizations, locations, or any other parts of the sentence.

⚠️ Do NOT modify:

Names of organizations or companies (e.g., Google, NASA, Mercadona)

Names of cities, towns, or countries (e.g., Madrid, Tokyo, Spain)

Articles, prepositions, or any structural elements of the sentence

✅ Only replace proper names of people with [MASK].

📏 IMPORTANT: The output must have exactly the same number of characters as the input sentence. Only replace person names with [MASK], and do not add or remove any other characters.

🛑 No words should be added or removed, and only the names of people should be replaced with [MASK].

Examples:

"John works at Google in the marketing department" → "[MASK] works at Google in the marketing department"

"Sarah is meeting with Michael in Tokyo" → "[MASK] is meeting with [MASK] in Tokyo"

"The project is a collaboration between Alice and Bob" → "The project is a collaboration between [MASK] and [MASK]"

Your turn: {text}"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",  # DeepSeek model
                messages=[{"role": "user", "content": input_text}],
                temperature=0.0,
                max_tokens=len(text) + 10  # Additional margin
            )
            result = response.choices[0].message.content.strip()
            
            # Basic length verification
            if abs(len(result) - len(text)) > 20:
                print(f"⚠️ Warning: Length difference ({len(text)}→{len(result)})")
            
            return result
        except Exception as e:
            print(f"⚠️ Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                print("❌ All attempts exhausted. Using original text.")
                return text

# Process all files
for file_path in datas_path:
    print(f"\n📂 Processing: {file_path}")
    df_original = pd.read_csv(file_path)
    
    # Reduction settings
    reduction_settings = {
        "news_4_anonimizar.csv": 0.90,
        "dbpedia_4_anonimizar.csv": 0.95,
        "cyber_4_anonimizar.csv": 0.90,
        "imdb_4_anonimizar.csv": 0.90
    }
    
    file_name = os.path.basename(file_path).lower()
    reduction_ratio = reduction_settings.get(file_name, 0.90)
    
    df_original_reduced, _ = train_test_split(
        df_original,
        test_size=reduction_ratio,
        random_state=42
    )
    
    df_anonymized = df_original_reduced.copy()

    for index in range(len(df_anonymized)):
        original_text = df_anonymized.iloc[index, 0]
        if pd.isna(original_text):
            print(f"Row {index}: null text, skipping.")
            continue
            
        anonymized_text = replace_entities_with_mask(str(original_text))
        df_anonymized.iloc[index, 0] = anonymized_text

        if index % 1 == 0:  # Show progress every row
            print(f"🔄 Progress: {index + 1}/{len(df_anonymized)} rows")
            
        time.sleep(0.1)  # Pause to avoid rate limits

    # Save results
    new_file_path = file_path.replace(".csv", "_PERanonimizado_DEEPSEEK.csv")
    df_anonymized.to_csv(new_file_path, index=False)
    print(f"✅ Saved: {new_file_path}")

print("\n🎉 Anonymization with DeepSeek completed!")