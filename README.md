# 📰 Fake News Detection Web App

This project is a machine learning-based web application that classifies news articles as **Fake** or **Real** using Natural Language Processing (NLP).

---

## 🚀 Project Overview

The goal of this project is to build a model that can automatically detect whether a news article is fake or real based on its textual content.

With the rapid spread of misinformation online, this type of system can help in identifying unreliable news sources and improving information credibility.

---

## ⚙️ Approach

The project follows a typical NLP pipeline:

1. **Data Collection**

   * Used a real-world dataset from Kaggle

2. **Data Preprocessing**

   * Combined fake and real news datasets
   * Labeled data (0 = Fake, 1 = Real)
   * Shuffled dataset for better training

3. **Feature Extraction**

   * Applied **TF-IDF Vectorization** to convert text into numerical features

4. **Model Training**

   * Used **Logistic Regression** for classification

5. **Evaluation**

   * Achieved approximately **98% accuracy**

6. **Deployment**

   * Built an interactive web app using **Streamlit**

---

## 📊 Model Performance

* Accuracy: ~98%
* The model performs well on clearly defined examples
* It may struggle with ambiguous or general statements due to reliance on word frequency rather than deep semantic understanding

---

## 🧠 Technologies Used

* Python
* Pandas
* Scikit-learn
* Streamlit
* Natural Language Processing (NLP)

---

## 📂 Dataset

This project uses the *Fake and Real News Dataset* from Kaggle.

The dataset consists of:

* **Fake.csv**: fake news articles
* **True.csv**: real news articles

Features include:

* Title
* Text
* Subject
* Date

Due to its large size, the dataset is not included in this repository.

Download it from:
https://www.kaggle.com/datasets/mdkalimullahzakir/fake-and-real-news-dataset

After downloading, place the files in the project directory:

* Fake.csv
* True.csv

---

## ▶️ How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/Fatimah2620/fake-news-detector.git
```

2. Install dependencies:

```bash
pip install streamlit pandas scikit-learn
```

3. Run the application:

```bash
streamlit run app.py
```

---

## 💻 Application Interface

The web application allows users to:

* Enter any news text
* Click on **Predict**
* Instantly receive whether the news is **Fake** or **Real**

---

## 💡 Future Improvements

* Use advanced models like BERT
* Improve text preprocessing
* Add confidence scores
* Deploy the app online

---

## 📌 Notes

* Dataset sourced from Kaggle
* Dataset excluded from repository due to size limitations
* This project is for educational and demonstration purposes

---

## 👩‍💻 Author

Fatimah
AI Student
