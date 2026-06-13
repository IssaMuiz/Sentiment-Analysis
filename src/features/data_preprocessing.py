import pandas as pd


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
    print(f"Dropped {initial_shape[0] - final_shape[0]} duplicated rows.")
    return df
