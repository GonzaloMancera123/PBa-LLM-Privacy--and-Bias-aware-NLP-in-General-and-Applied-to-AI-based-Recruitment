import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
from re import sub

# Initial configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"Using GPU: {gpus[0]}")
    except RuntimeError as e:
        print(e)

# Define list of masked dataset paths
masked_paths = [
    r'c:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\IMDB\GPT\4nano\IMDB_4_anonimizar_LOCanonimizado_CHATGPT4nano.csv',
    r'c:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\IMDB\GPT\4nano\IMDB_4_anonimizar_ORGanonimizado_CHATGPT4nano.csv',
    r'c:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\IMDB\GPT\4nano\IMDB_4_anonimizar_PERanonimizado_CHATGPT4nano.csv',
    r'c:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\IMDB\GPT\35\IMDB_4_anonimizar_LOCanonimizado_CHATGPT.csv',
    r'c:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\IMDB\GPT\35\IMDB_4_anonimizar_ORGanonimizado_CHATGPT.csv',
    r'c:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\IMDB\GPT\35\IMDB_4_anonimizar_PERanonimizado_CHATGPT.csv'
    # Add more paths here
]

# Load original data
df_original = pd.read_csv(r'C:\Users\Puesto-2\Documents\Gonzalo\NLP\ExtensionRevista\Databases\IMDB\IMDB Dataset.csv')

# Convert labels in the original dataset
df_original['sentiment'] = df_original['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Create train and test indices
train_idx, test_idx = train_test_split(df_original.index, test_size=0.2, shuffle=True, random_state=42)

# Create test splits (original data)
x_test = df_original.loc[test_idx, 'review'].copy()
y_test = df_original.loc[test_idx, 'sentiment'].copy()

# Text preprocessing
def preprocess_text(series):
    return series.apply(lambda x: x.lower()) \
                 .apply(lambda x: sub('([0-9])', '', x)) \
                 .apply(lambda x: sub(' +', ' ', x))

x_test = preprocess_text(x_test)

# Tokenization
maxlen = 200
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def tokenize(sentences, tokenizer):
    input_ids, input_masks = [], []
    for sentence in tqdm(sentences):
        inputs = tokenizer.encode_plus(
            sentence,
            max_length=maxlen,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=False
        )
        input_ids.append(inputs['input_ids'])
        input_masks.append(inputs['attention_mask'])    
    return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32')

x_test_ids, x_test_masks = tokenize(x_test, tokenizer)

# BERT Model
def create_model():
    bert = TFBertModel.from_pretrained('bert-base-uncased')
    input_ids = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32, name='input_ids')
    input_mask = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32, name='attention_mask')

    embeddings = bert(input_ids, attention_mask=input_mask)[0]
    out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
    out = tf.keras.layers.Dense(128, activation='relu')(out)
    out = tf.keras.layers.Dropout(0.1)(out)
    out = tf.keras.layers.Dense(32, activation='relu')(out)
    y = tf.keras.layers.Dense(1, activation='sigmoid')(out)

    model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
    for layer in model.layers:
        layer.trainable = True
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Training
results = []
for masked_path in masked_paths:
    print(f"\nTraining with file: {masked_path}")
    
    # Load masked data
    df_masked = pd.read_csv(masked_path)
    X_train = df_masked['texto'].copy()
    y_train = df_masked['etiquetas'].copy()
    
    # Preprocessing
    X_train = preprocess_text(X_train)
    X_train_ids, X_train_masks = tokenize(X_train, tokenizer)

    # Create and train the model
    model = create_model()
    history = model.fit(
        {'input_ids': X_train_ids, 'attention_mask': X_train_masks},
        y_train,
        batch_size=64,
        validation_data=({'input_ids': x_test_ids, 'attention_mask': x_test_masks}, y_test),
        epochs=10,
        verbose=1
    )
    
    # Training summary
    last_epoch = len(history.history['loss'])
    results.append({
        'masked_path': masked_path,
        'last_epoch': last_epoch,
        'final_loss': history.history['loss'][-1],
        'final_val_loss': history.history['val_loss'][-1],
        'final_accuracy': history.history['accuracy'][-1],
        'final_val_accuracy': history.history['val_accuracy'][-1]
    })

# Show summary
print("\nTraining summary:")
for result in results:
    print(f"File: {result['masked_path']}")
    print(f"Last epoch: {result['last_epoch']}")
    print(f"Final loss (training): {result['final_loss']:.4f}")
    print(f"Final loss (validation): {result['final_val_loss']:.4f}")
    print(f"Final accuracy (training): {result['final_accuracy']:.4f}")
    print(f"Final accuracy (validation): {result['final_val_accuracy']:.4f}\n")