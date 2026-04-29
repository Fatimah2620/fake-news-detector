#Import Library
import streamlit as st
import pandas as pd

# -----------------------------
# Load datasets (Fake & Real news)
# -----------------------------
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Assign labels: 0 = Fake, 1 = Real
fake["label"] = 0
true["label"] = 1

# Combine datasets into one
data = pd.concat([fake, true])

# Shuffle the data for better training
data = data.sample(frac=1)

# Split features and labels
X = data["text"]   # News content
y = data["label"]  # Target (Fake or Real)

# -----------------------------
# Convert text into numerical features using TF-IDF
# -----------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)
TfidfVectorizer(stop_words='english', max_df=0.7)

# -----------------------------
# Split dataset into training and testing sets
# -----------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train Logistic Regression model
# -----------------------------
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# Streamlit User Interface
# -----------------------------
st.title("📰 Fake News Detector")

# Text input from user
user_input = st.text_area("Enter news text:")

# Prediction button
if st.button("Predict"):
    vec = vectorizer.transform([user_input])
    prediction = model.predict(vec)

    # Display result
    if prediction[0] == 1:
        st.success("✅ Real News")
    else:
        st.error("❌ Fake News")