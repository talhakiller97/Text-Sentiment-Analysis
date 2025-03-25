# Text Sentiment Analysis

## Overview
This project is a **Text Sentiment Analysis System** that classifies movie reviews as **positive** or **negative** using **Natural Language Processing (NLP)** and **Machine Learning** techniques. It utilizes **Naive Bayes (MultinomialNB)** for classification.

## Dataset
The system is trained using the **IMDB Dataset** containing movie reviews and their corresponding sentiment labels (**positive/negative**).

## Features
- Text preprocessing using **NLTK** (stopword removal, lemmatization)
- **TF-IDF Vectorization** to transform text into numerical features
- **Naive Bayes Classifier** for sentiment prediction
- **Performance evaluation** using accuracy and classification report

## Installation
To run this project, install the required dependencies:

```bash
pip install pandas scikit-learn nltk
```

## Usage
1. Place the IMDB dataset (`IMDB Dataset.csv`) in the appropriate directory.
2. Run the Python script to preprocess the data, train the model, and evaluate its performance.
3. The output includes:
   - Preprocessed text samples
   - Model accuracy
   - Classification report

## Code Breakdown
1. **Data Loading:** Reads IMDB dataset into a Pandas DataFrame.
2. **Text Preprocessing:**
   - Converts text to lowercase
   - Removes stopwords
   - Applies lemmatization
3. **Feature Extraction:** Uses **TF-IDF Vectorizer** to transform text data.
4. **Model Training:**
   - Splits data into **training (70%)** and **testing (30%)** sets
   - Trains **Naive Bayes** classifier
5. **Evaluation:** Computes **accuracy** and displays a **classification report**.

## Model Performance
The model is evaluated based on:
- **Accuracy Score**
- **Precision, Recall, and F1-score**

## Future Improvements
- Use deep learning models (e.g., LSTMs, Transformers) for improved performance
- Implement real-time sentiment analysis using a web or CLI interface
- Extend dataset to include more diverse text sources

## Author
Developed by Talha Saeed

## License
This project is open-source and available for modification and improvement.

