# 📱 Mobile Review Sentiment Analysis — Simple RNN + NLTK

Predict customer sentiment (positive/negative) from mobile phone reviews using NLTK preprocessing + Simple RNN in TensorFlow/Keras.

## 🔎 Project Summary
- **Goal:** Binary sentiment classification
- **Input:** Product review text
- **Output:** Positive / Negative

## 🚀 Features
- Text cleaning + lemmatization
- Tokenization + padded sequences
- SimpleRNN + Embedding layers
- BatchNorm + Dropout regularization
- EarlyStopping callback

## 🛠 Tech Stack
Python, TensorFlow/Keras, NLTK, Pandas, Scikit-Learn, Matplotlib

## 📦 Installation

git clone <repo-link>
cd mobile-review-sentiment-rnn
pip install -r requirements.txt

## 🔧 Training
python train.py

## 🔮 Predict
from predict import predict_sentiment
predict_sentiment("Battery life is amazing!")

## 📊 Evaluation

Train/Validation accuracy curves

Train/Validation loss curves

## ✅ Improvements

LSTM/GRU

Transformers

API endpoint

## 📄 License
MIT License Copyright (c) 2025 AK-Jeevan. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

Feel free to fork, star, or contribute!
