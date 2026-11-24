# Fake News Detection using TF-IDF and Machine Learning

A beginner-friendly machine learning project that classifies a news article as **FAKE** or **REAL** using text features and an SGD-based linear classifier.

## Project overview

The notebook follows this workflow:

1. Load the news dataset from a CSV file.
2. Validate the expected `text` and `label` columns.
3. Remove rows with missing text or labels.
4. Split the data into training and testing sets (80/20).
5. Convert article text into TF-IDF features.
6. Train an `SGDClassifier` with hinge loss.
7. Evaluate predictions using accuracy and a confusion matrix.
8. Predict the label for a new news article.

## Repository structure

```text
fake-news-detection/
├── data/
│   └── .gitkeep
├── notebooks/
│   └── Fake_news_Detection.ipynb
├── src/
│   ├── train_model.py
│   └── predict.py
├── app.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset

The notebook expects a file named `news.csv` with these columns:

| Column | Description |
|---|---|
| `text` | Full news article or news text |
| `label` | Target class, expected to contain `FAKE` and `REAL` |

Download/use a dataset that you are permitted to use, then place it here:

```text
data/news.csv
```

The dataset is intentionally not committed. This avoids redistributing data without confirming its license and keeps large files out of Git.

## Setup

Python 3.9 or newer is recommended.

```bash
# Clone the repository
git clone https://github.com/<your-username>/fake-news-detection.git
cd fake-news-detection

# Create and activate a virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux activation:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Run the notebook

From the repository root:

```bash
jupyter notebook
```

Open `notebooks/Fake_news_Detection.ipynb` and run the cells in order. The notebook reads `../data/news.csv` relative to the notebook location.

## Run the Python script

```bash
python src/train_model.py --data data/news.csv
```

The script prints accuracy, a confusion matrix and a precision/recall/F1 report, then saves the model to `models/fake_news_model.joblib`.

## Predict and demo

```bash
python src/predict.py "Some news text"
streamlit run app.py
```

## Model and evaluation

- **Text representation:** `TfidfVectorizer` with English stop-word removal and `max_df=0.7`.
- **Classifier:** `SGDClassifier` using hinge loss.
- **Validation:** stratified 80/20 train-test split with `random_state=42`.
- **Metrics:** accuracy, confusion matrix, precision, recall and F1.

The notebook's classifier is an SGD hinge-loss classifier; it is not scikit-learn's `PassiveAggressiveClassifier` class.

## Important limitations

- Model performance depends on the dataset quality, size, class balance, and source coverage.
- A high test accuracy does not prove that a news story is true. The model learns patterns in the supplied training data and can make mistakes on new topics or sources.
- Use this as an educational text-classification project, not as a fact-checking authority.
- Review the dataset license and cite the dataset source in your project documentation.

## Future improvements

- Add ROC-AUC and cross-validation.
- Test on a separate, newer dataset to check generalization.
- Try other classifiers (Logistic Regression, Linear SVM) and compare.

## Author

**Arun Nawale**

If you use or extend this project, please update this section with your preferred profile and contact links.
