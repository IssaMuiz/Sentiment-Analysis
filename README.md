# Sentiment Analysis of Customer Reviews for a Movie


## Overview

This project is a Sentiment Analysis system built using Natural Language Processing (NLP) techniques and Machine Learning. The goal is to classify movie reviews as either positive or negative based on the text content.

The project follows a complete ML pipeline including data cleaning, text preprocessing, feature extraction using TF-IDF vectorization, and model training using Logistic Regression. The trained model can predict sentiment from unseen text inputs.

This project is designed for learning and practical experience in NLP and demonstrates how raw text data can be transformed into meaningful features for machine learning models. It also includes a simple structure that can be extended into a deployable web application using Streamlit.

## Key Features

* Text preprocessing (cleaning, tokenization, stopword removal, lemmatization)
* TF-IDF feature extraction
* Sentiment classification using Logistic Regression
* Model evaluation and accuracy measurement
* End-to-end NLP pipeline implementation
* Ready for deployment (Streamlit/FastAPI extension)

## Dataset
 or Text analytics.
 containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training and 25,000 for testing. So, predict the number of positive and negative reviews using either classification or deep learning algorithms.

The dataset used for this project is a IMDB dataset having 50,000 movie reviews for natural language processing. This is a dataset for binary sentiment classification. This dataset consists of 25,000 positive sentiment and 25,000 negative sentiment, it has two columns which are review and sentiment. The dataset can be downloaded using this link https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?utm_source=chatgpt.com


## Current Progress

### 1. Data Cleaning and Processing

* Data was loaded and cleaned, checking if there is any null values, and dropped the duplicated rows which are 418 rows, the dataset shape is now 49,582 columns and 2 rows
* Text preprocessing was performed using lowercasing, lemmatization, stopwords, and tokenization.
* Then another column ("cleaned_review") was created to store the preprocessed text and it was check side by side with the review column to confirm if the preprocess actually work.
* Another column was created ("encoded_sentiment") for sentiment encoding, 1 for positive and 0 for negative
* The dataset shape is now 49582 rows and 4 columns
* The dataset was splitted into training and testing set, 80/20 percent split was used, training set consists of 39665 rows and testing set consists of 9917 rows

### 2. Vectorization (TF-IDF)
* Term-frequency - Inverse Document Frequency known as TF-IDF was used as the vectorizatio method for this dataset and the max_features was set to 5000 words in order to reduce noise and avoid overfitting

### 3. Baseline Modelling (LogisticRegression)
* The baseline model was trained using LogisticeRegression
* The overall performance of the model is 88% accuracy which is a good headstart for the baseline


Then we have the full classification as:
| Metric   | Positive  | Negative  |
| -------- | --------- | ----------|
| Precision| 0.87      |  0.89     |
| Recall   | 0.90      |  0.87     |



### 4. Error Analysis
The confusion matrix was observed:
                 

     Predicted  Negative  Positive

Actual Negative  4287      652

Actual Positive   513      4465

* The total predicted reviews are 9917 reviews
* The model is making mistake of False postive of 652 reviews, negative reviews predicted as positive review
* False negative of 513 reviews, positive reviews predicted as negative reviews
* The total correct prediction is 8752 reviews
* The total wrong prediction is 1165 reviews
* Improvement to this model will be conducted and evaluated in the next version

## Project Steps

- Data collection and loading
- Data cleaning and text preprocessing
- Data splitting
- Vectorization (TF-IDF)
- Baseline modeling
- Error analysis
- Model Pipeline
- Artifact creation
- Model packaging and deployment readiness
- Baseline model deployment
- hyperparameter tuning and model optimization
- Final evaluation
- Unit testing
- New model deploymnent
- Monitoring and maintenance plan



## Contribution

This is currently a personal MVP project. Contributions, ideas, and feedback are welcome.



## 👤 Author

Built by Issa Muiz
Machine Learning & Deep Learning Enthusiast
