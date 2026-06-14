import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


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
