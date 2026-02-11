# 📩 SMS Spam Classifier

A Machine Learning-based web application that classifies SMS messages as **Spam 🚨** or **Ham ✅ (Not Spam)** using Natural Language Processing (NLP) and a Multinomial Naive Bayes model.

---

## 🌐 Live Demo

👉 **Try it here:**  
https://sms-spam-classifier101.streamlit.app/

---

## 🔥 Features

-  Real-time SMS Classification – Enter any message and get instant predictions.
-  User-friendly Streamlit Interface – Simple, intuitive, and interactive.
-  Machine Learning Powered – Uses NLP preprocessing and TF-IDF vectorization.
-  Detects spam messages like:
  - "Congratulations! You have won ₹5000 cash. Claim now!!!"
  - "Free entry in a weekly contest. Call 09061701461 now!"
  - "Hey, you won’t believe this deal I found online!"
- ✅ Handles normal messages like:
  - "Call me when you reach home."
  - "Let's grab lunch tomorrow."

---

## 🛠️ Technologies Used

- Python – Core programming language  
- Pandas & NumPy – Data cleaning and manipulation  
- Scikit-learn:
  - TF-IDF Vectorizer  
  - Train/Test Split  
  - Multinomial Naive Bayes Classifier  
  - Evaluation Metrics  
- Joblib – Save and load trained model  
- Streamlit – Web application interface  

---

## 📊 Dataset

Dataset Used: `spam.csv`

### Columns:
- `label` – "ham" (normal message) or "spam" (unwanted message)
- `message` – SMS text content

### Preprocessing Steps:
- Removed unnecessary columns  
- Renamed columns to `label` and `message`  
- Converted labels to numeric:
  - ham = 0  
  - spam = 1  

---

## 📈 Model Training

- Vectorization: TF-IDF Vectorizer  
- Train/Test Split: 80% training, 20% testing  
- Model: Multinomial Naive Bayes  

### 📊 Performance

- Accuracy: 96.23%  
- Precision, Recall, F1-score: Above 0.85 for both spam and ham  

### 💾 Saved Files

- `spam_model.pkl`  
- `vectorizer.pkl`  

---

## 🛠️ How It Works

1. User enters an SMS message in the Streamlit app.
2. The message is transformed using the saved TF-IDF vectorizer.
3. The trained Multinomial Naive Bayes model predicts the label.
4. The result is displayed as:
   - Spam 🚨  
   - Ham ✅  

---

## 💻 Example Usage (Python)

```python
from joblib import load

# Load saved model and vectorizer
model = load("spam_model.pkl")
vectorizer = load("vectorizer.pkl")

def predict_message(message):
    vectorized = vectorizer.transform([message])
    prediction = model.predict(vectorized)[0]
    return "Spam 🚨" if prediction == 1 else "Ham ✅"

# Example predictions
print(predict_message("Congratulations! You won a free ticket. Call now!"))
print(predict_message("Call me when you reach home."))
