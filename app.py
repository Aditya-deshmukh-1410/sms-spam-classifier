import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📩 SMS Spam Classifier")

message = st.text_area("Enter your message to check whether it's Spam or Ham:")

if st.button("Predict"):
    if message.strip() != "":
        vectorized = vectorizer.transform([message])
        
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]

        spam_prob = probability[1] * 100
        ham_prob = probability[0] * 100

        if prediction == 1:
            st.error(f"🚨 Spam ({spam_prob:.2f}% confidence)")
        else:
            st.success(f"✅ Ham ({ham_prob:.2f}% confidence)")

        st.write("### Confidence Scores")
        st.write(f"Ham: {ham_prob:.2f}%")
        st.write(f"Spam: {spam_prob:.2f}%")
