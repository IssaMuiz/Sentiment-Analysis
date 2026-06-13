import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(df, test_size=0.3, random_state=42):
    """
    Split the DataFrame into training, validation, and testing sets.
    Args:
        df (pd.DataFrame): The input DataFrame to be split.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before applying the split.
    Returns: tuple: A tuple containing the training, validation, and testing DataFrames.
    """

    train_df, test_val_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    val_df, test_df = train_test_split(
        test_val_df, test_size=0.5, random_state=random_state
    )

    print(f"Training set size: {train_df.shape[0]} samples")
    print(f"Validation set size: {val_df.shape[0]} samples")
    print(f"Testing set size: {test_df.shape[0]} samples")
    return train_df, val_df, test_df


def split_features_and_target(df: pd.DataFrame):
    """
    Split the DataFrame into features (X) and target variable (y).
    Args:
        df (pd.DataFrame): The input DataFrame containing the features and target variable.
    Returns: tuple: A tuple containing the features DataFrame (X) and target variable Series (y).
    """
    X = df["cleaned_review"]
    y = df["encoded_sentiment"]
    return X, y
