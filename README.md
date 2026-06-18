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
* The dataset was further splitted into train, validation and test set, hyperparameter tuning will be done on the validation set to avoid data leakage, and the test set will be use for the final evaluation
* The train set consists of 34707 rows (70% of the dataset), validation set consists of 7437 rows (15% of the dataset) and the test set consists of 7438 rows (15% of the dataset)

### 2. Vectorization (TF-IDF)
* Term-frequency - Inverse Document Frequency known as TF-IDF was used as the vectorization method for this dataset and the max_features was set to 5000 words, ngram_range(1, 2) in order to reduce noise, avoid overfitting and better generalization

### 3. Baseline Modelling (LogisticRegression)
* The baseline model was trained using LogisticeRegression
* The overall performance of the model is 88% accuracy which is a good headstart for the baseline
* The roc-auc score was added to the evaluation metrics, 95% roc-auc was recorded, which indicates the model perform great across all thresholds


Then we have the full classification as:
| Metric   | Positive  | Negative  |
| -------- | --------- | ----------|
| Precision| 0.87      |  0.89     |
| Recall   | 0.90      |  0.87     |



### 4. Error Analysis
The confusion matrix was observed:

* The total predicted reviews are 7437 reviews
* The model is making mistake of False postive of 476 reviews, negative reviews predicted as positive review
* False negative of 382 reviews, positive reviews predicted as negative reviews
* The total correct prediction is 6579 reviews
* The total wrong prediction is 858 reviews


### 5. Model pipeline building
This project uses a Scikit-Learn Machine Learning Pipeline to streamline the entire sentiment analysis workflow from raw text to prediction.

The pipeline combines all preprocessing and modeling steps into a single reusable object.

- Pipeline Steps
  
1. Text Preprocessing
 - Cleaning of raw text (lowercasing, removing noise, etc.)
 - Implemented using a custom preprocessing function wrapped with FunctionTransformer

2. Feature Extraction
 - TF-IDF Vectorization
 - Converts text into numerical features

3. Model Training
 - Logistic Regression classifier
 - Trained on TF-IDF features 

### Model Artifacts & Registry

This project uses a Model Registry system to manage trained models and their associated metadata in a structured and reproducible way.

After training, the model is saved as a single deployable artifact along with its evaluation metrics and training configuration.

- What gets saved as an artifact

 - Each model version includes:
     1. Trained ML Pipeline (Preprocessing + TF-IDF + Classifier)
     2. Evaluation Metrics (Accuracy, Precision, Recall, F1-score)
     3. Model Metadata (model type, version, dataset info, training date)


### Model Comparison and Selection

| Model               | Train Score | Accuracy  | F1        | ROC-AUC   | Generalization   |
| ------------------- | ----------- | --------- | --------- | --------- | ---------------- |
| Linear SVC          | 0.934       | 0.878     | 0.878     | 0.948     | Good             |
| Logistic Regression | 0.913       | **0.888** | **0.888** | **0.954** | **Best balance** |
| Multinomial NB      | 0.866       | 0.853     | 0.853     | 0.929     | Weakest          |
| Random Forest       | 1.000       | 0.846     | 0.846     | 0.922     | Overfitting      |

* This is the comparison analysis for all the model used for the sentiment analysis dataset training. The models were compared using Accuracy, Precision, Recall, F1-score, and ROC-AUC on the validation set.

* **Logistic Regression** achieved the best overall performance with the highest accuracy (0.888) and ROC-AUC (0.954), showing strong generalization.
* **Linear SVC** performed closely behind with slightly lower accuracy but a strong ROC-AUC score (0.947).
* **Multinomial Naive Bayes** provided a solid baseline but lagged behind linear models in overall performance.
* **Random Forest** showed clear overfitting and performed the weakest on this text-based TF-IDF representation.

* Logistic Regression was selected as the final model due to its best balance of performance, stability, and efficiency for deployment.

* Hyperparameter tuning and model optimization will be done using the selected model (Logistic Regression) 


### Hyperparameter tuning
* Hyperparameter tuning was performed using GridSearchCV on the best-performing model (Logistic Regression) to optimize performance. The search focused on key parameters such as C, TF-IDF ngram_range, and max_features, using cross-validation for reliable evaluation.

* Although tuning confirmed stable performance (AUC-ROC 0.953910382017343), it did not provide a significant improvement over the baseline model (AUC-ROC 0.954). The final selected model remains the baseline Logistic Regression due to its strong and consistent results. 

### Unit Testing

Unit tests were implemented using Pytest to validate key components of the machine learning pipeline. Tests cover text preprocessing, model training, prediction, and model registry functionality to ensure the system behaves as expected and remains reliable during development and deployment.



## Project Steps

- Data collection and loading
- Data cleaning and text preprocessing
- Data splitting
- Vectorization (TF-IDF)
- Baseline modeling
- Error analysis
- Model Pipeline
- Artifact creation
- models comparison and selection
- hyperparameter tuning and model optimization
- Unit testing
- Model packaging and deployment
- Monitoring and maintenance plan



## Contribution

This is currently a personal MVP project. Contributions, ideas, and feedback are welcome.



## 👤 Author

Built by Issa Muiz
Machine Learning & Deep Learning Enthusiast
