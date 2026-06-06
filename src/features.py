import pandas as pd
from src.text_preprocessing import preprocess_text


def cleaned_review(df: pd.DataFrame):
    """
    Add a new column 'cleaned_review' to the DataFrame by applying the
        preprocess_text function to the 'review' column.
    Args:
        df (pd.DataFrame): The input DataFrame containing a 'review' column.
    Returns:
        pd.DataFrame: The DataFrame with an additional 'cleaned_review' column.
    """

    df["cleaned_review"] = df["review"].apply(preprocess_text)
    print("Added 'cleaned_review' column to the DataFrame.")
    return df


def encode_sentiment(df: pd.DataFrame):
    """
    Encode the 'sentiment' column in the DataFrame to numerical values.
    Args:
        df (pd.DataFrame): The input DataFrame containing a 'sentiment' column.
    Returns: pd.DataFrame: The DataFrame with the 'sentiment' column encoded
        as numerical values.
    """
    sentiment_mapping = {"positive": 1, "negative": 0}
    df["encoded_sentiment"] = df["sentiment"].map(sentiment_mapping)
    print(
        "Encoded 'sentiment' column to 'encoded_sentiment' with values 1 for positive and 0 for negative."
    )
    return df
