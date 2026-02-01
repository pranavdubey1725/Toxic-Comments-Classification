# Toxic Comment Classification (NLP)

## OVERVIEW
- Builds an end-to-end **NLP pipeline** to classify toxic comments using both traditional machine learning models and deep learning (LSTM).
- Compares classical text models vs neural networks on a real-world dataset to understand trade-offs between performance and generalization.
- Addresses the challenges of working with **highly imbalanced datasets** in text classification tasks.

---

## PROJECT STRUCTURE
- **ToxicCommentClassification.ipynb** – Main notebook for data preprocessing, text cleaning, model training, and evaluation.
- **train.csv** – Dataset containing comment text and toxicity labels *(not included due to size and licensing restrictions)*.
- **requirements.txt** – A list of the project's Python dependencies.

---

## DATASET USED
- **Source:** [Kaggle – Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- **Total samples:** ~160,000 comments
- **Class distribution:** Highly imbalanced dataset with toxic comments representing approximately **10%** of the total data
- **Note:** Dataset is not included in this repository due to size and licensing restrictions. Download it from the Kaggle link above.

---

## TECH STACK
- **Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib
- **Techniques:** Text Preprocessing, TF-IDF Vectorization, Stopword Removal, Tokenization, Sequence Padding, Deep Learning (LSTM), Classification Modeling
- **Environment:** Jupyter Notebook

---

## NLP PIPELINE
The following preprocessing steps were applied to clean and prepare the text data:
- **Lowercasing** – Converting all text to lowercase for consistency
- **URL Removal** – Removing web links from comments
- **Punctuation Removal** – Stripping special characters
- **Stopword Removal** – Removing common words that don't contribute to toxicity classification
- **Tokenization** – Breaking text into individual tokens
- **Padding Sequences** – Ensuring uniform input length for LSTM model

---

## MODELS USED
Three different approaches were implemented and compared:

1. **TF-IDF + Naive Bayes** – Traditional probabilistic classifier
2. **TF-IDF + Logistic Regression** – Linear model with better handling of class imbalance
3. **LSTM (Keras/TensorFlow)** – Deep learning approach for sequence modeling

---

## MODEL PERFORMANCE

### TF-IDF + Naive Bayes
- **Accuracy:** ~94.9%
- **Toxic class recall:** ~51%
- **F1-score (toxic):** ~0.66

### TF-IDF + Logistic Regression
- **Accuracy:** ~91.6%
- **Toxic class recall:** ~84%
- **Precision (toxic):** ~54%
- **Better balance** between recall and false positives

### LSTM Model
- **Training accuracy:** ~97.7%
- **Validation accuracy:** ~95.3%
- **Overfitting observed** – Validation loss increased with epochs
- **Best performance achieved with:**
  - Epochs: 3–5
  - Batch size: 64
  - Dropout regularization

---

## KEY OBSERVATIONS
- Classical models with **TF-IDF performed strongly** on this dataset despite their simplicity
- **Logistic Regression** handled class imbalance better than Naive Bayes, achieving higher recall on toxic comments
- **LSTM** learned rich text representations but **overfit quickly** without proper regularization
- **Validation metrics** were more reliable than training accuracy for assessing true model performance
- The trade-off between model complexity and generalization is evident: simpler models generalized better on this task

---

## HOW TO RUN

1. **Download the dataset from Kaggle**
   - Visit: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
   - Download the dataset files

2. **Clone the repository**
```bash
   git clone <repository-url>
   cd <repository-name>
```

3. **Place dataset files in the project directory**
```bash
   # Ensure train.csv is in the same directory as the notebook
```

4. **Create a virtual environment and install dependencies**
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
```

5. **Launch the Jupyter Notebook**
```bash
   jupyter notebook
```
   Open and run the cells within **ToxicCommentClassification.ipynb** to execute the NLP pipeline and model training.

---

## AUTHOR
**Pranav Dubey**

---

## LICENSE
This project is for educational purposes. The dataset is subject to Kaggle's terms and conditions.
