# Twitter Sentiment Analysis Machine Learning Project

This repository contains a complete end-to-end Machine Learning pipeline for sentiment analysis using the **Twitter Entity Sentiment Analysis** dataset. The project compares multiple classical ML algorithms to classify tweets into four sentiment categories.

##  Project Overview

The goal of this project is to build and evaluate models that can automatically detect the sentiment of a tweet (Positive, Negative, Neutral, or Irrelevant) toward specific topics or entities.

##  Dataset

The project uses the [Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis) dataset from Kaggle.

* **Training Set:** ~75,000 records
* **Validation Set:** 1,000 records
* **Labels:** Positive, Negative, Neutral, Irrelevant

##  Tech Stack

* **Language:** Python 3.x
* **Core Libraries:** `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`
* **NLP:** `NLTK` (Tokenization, Stopwords removal, Lemmatization)
* **ML Framework:** `Scikit-learn`
* **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)

##  Pipeline Steps

1. **Preprocessing:** Text cleaning (lowercase, URL/mention removal), NLTK tokenization, stopword removal, and WordNet lemmatization.
2. **Feature Engineering:** Converting text into numerical vectors using `TfidfVectorizer` (with bigrams).
3. **Model Comparison:** Evaluating five different classifiers:
* Logistic Regression
* Naive Bayes (Multinomial)
* Decision Tree
* Random Forest
* Linear SVM


4. **Evaluation:** Comparing models based on Accuracy, Precision, Recall, and F1-Score.

##  Results

The models were evaluated on the validation set with the following accuracy results:

| Model | Accuracy |
| --- | --- |
| **Random Forest** | **94.60%** |
| Linear SVM | 90.40% |
| Decision Tree | 88.60% |
| Logistic Regression | 86.50% |
| Naive Bayes | 74.50% |

##  Installation & Usage

1. **Clone the repo:**
```bash
git clone https://github.com/your-username/twitter-sentiment-analysis.git

```


2. **Install dependencies:**
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn

```


3. **Run the Notebook:**
Open `sent-analysis-main.ipynb` in Jupyter or Google Colab to see the full analysis and training process.
