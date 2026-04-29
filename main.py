#Import Library
import pandas as pd

# Upload Data
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add label
fake["label"] = 0
true["label"] = 1

# Data merging
data = pd.concat([fake, true])

# خلط البيانات
data = data.sample(frac=1)

# The Text
X = data["text"]
y = data["label"]

# -------------------------
# Convert Text to Number
# -------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

# -------------------------
# Siplit Data
# -------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Training The model
# -------------------------
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------
# Evaluation
# -------------------------
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

def predict_news(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)
    return "Real" if pred[0] == 1 else "Fake"


# Test
sample = "Breaking: Scientists discover new cure for cancer"
print("Prediction:", predict_news(sample))