# 📊 YouTube Comment Spam Detection using Ensemble Learning

## 📌 Overview

YouTube’s comment sections are often flooded with spam that reduces meaningful interaction. This project implements a **spam detection model** using **ensemble learning** by combining **Logistic Regression, Multinomial Naive Bayes, and Support Vector Machine (SVM)**.

The pipeline covers:

* Data loading and preprocessing
* Feature extraction using **TF-IDF**
* Model training with an **ensemble classifier**
* Evaluation with multiple metrics and visualizations

The trained model achieves an accuracy of **\~73%**, providing a strong baseline for further improvements.



## 📂 Project Structure


├── Spam_Detection_Ensemble.ipynb         # Main notebook: preprocessing, modeling, evaluation
├── YoutubeCommentsDataSet.csv            # Cleaned dataset of YouTube comments
├── youtube-comments-dataset-metadata.json # Metadata (source, license, dataset description)
├── requirements.txt                      # Dependencies (to be generated)
└── README.md                             # Project documentation



## 📊 Dataset

* **Source**: [Kaggle - YouTube Comments Dataset](https://www.kaggle.com/datasets/atifaliak/youtube-comments-dataset/versions/1)
* **Creator**: Atif Ali AK
* **License**: [Open Database License (ODbL 1.0)](http://opendatacommons.org/licenses/dbcl/1.0/)&#x20;
* **Contents**:

  * `Comment` → Raw YouTube comment text
  * `Sentiment` → Label (`Positive`, `Negative`, or `Neutral`)

The dataset is **fully cleaned and preprocessed**, making it suitable for **spam detection, sentiment analysis, topic modeling, and user behavior analysis**.


## 🚀 Features

✔️ Text preprocessing (cleaning, stopword removal, stemming)
✔️ Feature extraction using **TF-IDF vectorization**
✔️ Soft voting **ensemble classifier** with:

* Logistic Regression
* Multinomial Naive Bayes
* Linear SVM
  ✔️ Evaluation with:
* Accuracy score
* Precision, Recall, F1-score
* Confusion matrix heatmap
* ROC curve and AUC


## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/parulgoel20/YouTube-Comment-Spam-Detection-using-Ensemble-Learning.git
cd YouTube-Comment-Spam-Detection-using-Ensemble-Learning
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies include:

* pandas
* numpy
* scikit-learn
* nltk
* matplotlib
* seaborn


## ▶️ Usage

1. Place your dataset as `YoutubeCommentsDataSet.csv` in the project root (already included here).
2. Run the Jupyter Notebook:

```bash
jupyter notebook Spam_Detection_Ensemble.ipynb
```

3. The notebook will:

   * Preprocess the comments
   * Train the ensemble classifier
   * Output evaluation results and visualizations


## 📊 Results

* **Accuracy**: \~73%
* **Classification Report**: Balanced performance across classes
* **Confusion Matrix**: Highlights false positives/negatives
* **ROC Curve & AUC**: Measures overall discriminative power

While 73% accuracy is a solid baseline, this model can be significantly improved with tuning and advanced architectures.


## 📌 Methodology

1. **Preprocessing**

   * Lowercasing
   * Removing URLs, special characters, and stopwords
   * Stemming with NLTK

2. **Feature Engineering**

   * TF-IDF vectorization (unigrams, max\_df=0.7, English stopwords)

3. **Modeling**

   * Logistic Regression (max\_iter=1000)
   * Multinomial Naive Bayes
   * Linear SVM with probability enabled
   * Ensemble using **VotingClassifier (soft voting)**

4. **Evaluation**

   * 70/30 train-test split (stratified)
   * Metrics: Accuracy, classification report, confusion matrix, ROC curve


## 📌 Future Improvements

🔹 Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
🔹 Try additional models (Random Forest, XGBoost, LightGBM)
🔹 Experiment with **deep learning models** (LSTMs, BERT, DistilBERT)
🔹 Deploy model via **Flask/FastAPI API** or streamlit web app for real-time spam detection
🔹 Add cross-validation and learning curve analysis


## 📜 License

* **Code**: Licensed under the **MIT License**
* **Dataset**: Licensed under [ODbL 1.0](http://opendatacommons.org/licenses/dbcl/1.0/)&#x20;


✨ This project provides a solid foundation for **YouTube comment spam detection** and can be extended into advanced NLP research or production-ready systems.
