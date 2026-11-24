import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
df = pd.read_csv("news.csv")
df
print("Shape:", df.shape)
print(df.head())

#  2. Validate expected columns 
required_cols = {"text", "label"}
missing = required_cols - set(df.columns.str.lower())
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}. "
                     f"Found: {list(df.columns)}")

# Normalise column names to lowercase
df.columns = df.columns.str.lower()

#  3. Drop nulls & inspect label distribution 
df.dropna(subset=["text", "label"], inplace=True)
print("\nLabel distribution:\n", df["label"].value_counts())
#  4. Features & labels 
X = df["text"]
y = df["label"]

#  5. Train / test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  6. TF-IDF vectorisation 
tfidf = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

#  7. Train PassiveAggressiveClassifier 
pac = SGDClassifier(loss="hinge", penalty=None, learning_rate="pa1", eta0=1.0, max_iter=50, random_state=42)
pac.fit(X_train_tfidf, y_train)

#  8. Evaluate 
y_pred = pac.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
cm       = confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"])

print(f"\nAccuracy : {round(accuracy * 100, 2)}%")
print("\nConfusion Matrix (rows=actual, cols=predicted):")
print(pd.DataFrame(cm,
                   index=["Actual FAKE", "Actual REAL"],
                   columns=["Pred FAKE",  "Pred REAL"]))

#  9. Predict on new text
def predict_news(text: str) -> str:
    """Return 'FAKE' or 'REAL' for a given news article string."""
    vec = tfidf.transform([text])
    return pac.predict(vec)[0]
    # Quick demo
sample = "Scientists discover water on Mars, government confirms."
print(f"\nSample prediction → '{sample}'\nResult: {predict_news(sample)}")
