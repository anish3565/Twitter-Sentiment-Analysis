# 🐦 Twitter Sentiment Analysis

This project analyzes the sentiment of tweets using Natural Language Processing (NLP) and machine learning techniques. It classifies tweets into **positive**, **neutral**, or **negative** sentiments by preprocessing tweet data and training classification models.

<br>

## 📁 Project Structure

```
Twitter-Sentiment-Analysis/
│
├── Twitter Sentiment Analysis.ipynb  # Jupyter Notebook with full workflow
├── Twitter Sentiments.csv            # Dataset with tweet text and sentiment labels
├── .gitignore                        # Git ignore rules
└── README.md                         # Project documentation
```

---

## 💡 Objectives

- Load and explore the Twitter sentiment dataset.
- Preprocess tweet text: remove stop words, links, mentions, etc.
- Visualize sentiment distribution and key insights.
- Train multiple machine learning models for sentiment classification.
- Evaluate and compare model performance.

---

## 📊 Dataset Overview

**File:** `Twitter Sentiments.csv`

Each record in the dataset contains:

- **Tweet**: The tweet text.
- **Sentiment**: Label indicating the sentiment (`positive`, `neutral`, or `negative`).

**Sample:**

| Tweet                            | Sentiment  |
|----------------------------------|------------|
| "I love sunny days!"            | positive   |
| "This is just okay."            | neutral    |
| "I hate waiting in traffic."    | negative   |

---

## ⚙️ Installation

1. **Clone the repository:**

```bash
git clone https://github.com/anish3565/Twitter-Sentiment-Analysis.git
cd Twitter-Sentiment-Analysis
```

2. **Set up a virtual environment (optional):**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install required dependencies:**

Create a `requirements.txt` (if not present) with packages like:

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
nltk
```

Then install with:

```bash
pip install -r requirements.txt
```

4. **Download NLTK resources:**

Within the notebook, or separately:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

---

## 🧠 Machine Learning Workflow

The full pipeline is implemented in the notebook:

### 1. Data Cleaning & Preprocessing

- Convert to lowercase
- Remove punctuation, links, mentions, hashtags
- Tokenization
- Remove stop words
- Stemming (optional)

### 2. Exploratory Data Analysis (EDA)

- Count of sentiments
- WordClouds per sentiment
- Frequent word distributions

### 3. Feature Extraction

- TF-IDF Vectorization using `TfidfVectorizer` from `sklearn`.

### 4. Model Training

Algorithms used:

- Logistic Regression
- Naive Bayes
- Support Vector Machines (SVM)

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
```

### 5. Model Evaluation

Metrics used:

- Accuracy
- Confusion Matrix
- Classification Report

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

---

## 📈 Results

- Accuracy varies by model and parameters, typically between **75% - 85%**.
- SVM generally performs well on text classification tasks.
- Data cleaning has a significant impact on performance.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit your changes
4. Push and create a Pull Request

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).

---

## 📬 Contact

Created by [Anish](https://github.com/anish3565) – feel free to connect!

---
