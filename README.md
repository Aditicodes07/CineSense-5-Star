# 🎬 CineSense 5★

Multi-Class Sentiment Analysis for Movie Reviews using NLP.

## Project Structure

```
CineSense-5-Star/
├── assets/
├── data/
│   └── movie_reviews.csv
├── models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
├── app.py
├── preprocess.py
├── train_model.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Train the Model

```bash
python train_model.py
```

## Run the App

```bash
streamlit run app.py
```

## Tech Stack

- Python 3.13
- Streamlit — UI
- scikit-learn — Logistic Regression + TF-IDF
- NLTK — Text preprocessing
- Plotly — Probability chart

## How It Works

1. User enters a movie review
2. Text is preprocessed (lowercase, stopword removal, lemmatization)
3. TF-IDF vectorizer converts text to features
4. Logistic Regression predicts rating (1★ – 5★)
5. Results displayed with confidence score and probability chart
