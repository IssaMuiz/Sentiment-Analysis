from pathlib import Path
from src.features.data_preprocessing import load_data, drop_duplicates
from src.features.build_features import cleaned_review, encode_sentiment
from src.features.split_data import split_data, split_features_and_target

# Define paths for raw and processed data
RAW_DATA_PATH = Path("data/raw/IMDB Dataset.csv")
PROCESSED_DATA_PATH = Path("data/processed/")
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)


# Load raw data
df = load_data(RAW_DATA_PATH)

# Drop duplicates
df = drop_duplicates(df)

# Preprocess text data
df = cleaned_review(df)
df = encode_sentiment(df)


# Split data into training, validation and testing sets
train_df, val_df, test_df = split_data(df)
X_train, y_train = split_features_and_target(train_df)
X_val, y_val = split_features_and_target(val_df)
X_test, y_test = split_features_and_target(test_df)

# Save processed data
X_train.to_csv(PROCESSED_DATA_PATH / "X_train.csv", index=False)
y_train.to_csv(PROCESSED_DATA_PATH / "y_train.csv", index=False)
X_val.to_csv(PROCESSED_DATA_PATH / "X_val.csv", index=False)
y_val.to_csv(PROCESSED_DATA_PATH / "y_val.csv", index=False)
X_test.to_csv(PROCESSED_DATA_PATH / "X_test.csv", index=False)
y_test.to_csv(PROCESSED_DATA_PATH / "y_test.csv", index=False)


print(
    "Data preparation completed successfully. "
    "Processed data saved to 'data/processed/' directory."
)
