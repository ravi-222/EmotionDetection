import pandas as pd

# Adjust the path to where you've saved the DailyDialog dataset
dataset_path = 'path/to/your/dailydialog.csv'

# Load the dataset
dialog_df = pd.read_csv(dataset_path)

# Quick exploration
print(dialog_df.head())
print("\nDataset size: ", dialog_df.shape)
print("\nEmotion distribution:\n", dialog_df['emotion'].value_counts())


import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_text(text):
    """Clean and preprocess text for modeling."""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[\r|\n|\r\n]+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

# Apply preprocessing
dialog_df['dialog'] = dialog_df['dialog'].apply(preprocess_text)

# Tokenization and padding
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(dialog_df['dialog'])
sequences = tokenizer.texts_to_sequences(dialog_df['dialog'])
padded_sequences = pad_sequences(sequences, padding='post')

# Since this is unsupervised, we won't use train_test_split for splitting but rather show how it can be done
X_train, X_val, _, _ = train_test_split(padded_sequences, dialog_df['dialog'], test_size=0.2, random_state=42)


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.layers import Dense, Embedding

max_sequence_length = padded_sequences.shape[1]
embedding_dim = 50
latent_dim = 100  # Latent dimensionality of the encoding space.

# Define an encoder-decoder model
input_seq = Input(shape=(max_sequence_length,))
embedded_seq = Embedding(input_dim=10000, output_dim=embedding_dim, input_length=max_sequence_length)(input_seq)
encoded = LSTM(latent_dim)(embedded_seq)
decoded = RepeatVector(max_sequence_length)(encoded)
decoded = LSTM(embedding_dim, return_sequences=True)(decoded)
decoded = Dense(10000, activation='softmax')(decoded)

autoencoder = Model(input_seq, decoded)
autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

print(autoencoder.summary())


autoencoder.fit(X_train, X_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(X_val, X_val))
