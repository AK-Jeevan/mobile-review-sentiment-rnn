# This Neural Network classifies mobile phone reviews as Positive or Negative using SimpleRNN and NLTK for text preprocessing.

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\akjee\Documents\AI\NLP\NLP - DL\RNN\Mobile_Reviews.csv")
df = df[['review_text', 'sentiment']].dropna().reset_index(drop=True)
df['label'] = df['sentiment'].map({'Positive': 1, 'Negative': 0})
df = df.dropna(subset=['label']).reset_index(drop=True)

# 🧹 NLTK preprocessing
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')

tokenizer_nltk = TreebankWordTokenizer()
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = tokenizer_nltk.tokenize(str(text).lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    stemmed = [stemmer.stem(t) for t in tokens]
    lemmatized = [lemmatizer.lemmatize(t) for t in stemmed]
    return ' '.join(lemmatized)

df['clean_text'] = df['review_text'].apply(preprocess)

# 🔠 Keras Tokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['clean_text'])
sequences = tokenizer.texts_to_sequences(df['clean_text'])

max_len = 100
X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
y = df['label'].values

# 🔀 Train-test split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 🧠 Build RNN model with regularization
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

vocab_size = 10000

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=32, input_length=max_len),
    SimpleRNN(64, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 🚀 Train with early stopping
early = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, callbacks=[early])

# 📊 Plot accuracy and loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy Curve')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss Curve')
plt.tight_layout()
plt.show()

# 🔍 Prediction helper
def predict_sentiment(text):
    cleaned = preprocess(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    pred = model.predict(padded)[0][0]
    return "Positive" if pred > 0.5 else "Negative"

# 🧪 Examples
print(predict_sentiment("Battery life is amazing and camera is stunning!"))
print(predict_sentiment("The phone is slow and the screen is dull."))