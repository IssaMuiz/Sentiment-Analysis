import pandas as pd

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def load_data(file_path):
    """
    Load data from a CSV file.
    Args:
        file_path (str): The path to the CSV file.
    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from ({file_path})")
        return data
    except Exception as e:
        print(f"Error loading data from ({file_path}): {e}")
        return None


def drop_duplicates(df):
    """
    Drop duplicate rows from the DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        pd.DataFrame: The DataFrame with duplicate rows removed.

    """
    initial_shape = df.shape
    df = df.drop_duplicates()
    final_shape = df.shape
    print(f"Dropped {initial_shape[0] - final_shape[0]} duplicate rows.")
    return df


def preprocess_text(text):

    # lowercase
    text = text.lower()

    # remove html tags
    text = re.sub(r"<.*?>", "", text)

    # remove non-letters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # tokenize
    words = text.split()

    # remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(words)  # Join the list of words back into a single string
