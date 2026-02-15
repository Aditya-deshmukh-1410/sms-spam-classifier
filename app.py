import streamlit as st
import joblib
import numpy as np
import time
import random


# ---------------- LOAD ML MODEL ----------------
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI-Powered SMS Spam Detector",
    page_icon=" ⚛ ",
    layout="centered"
)

st.title(" ✦ AI-Powered SMS Spam Detector")
st.caption("Machine Learning + Explainable Artificial Intelligence")

# ---------------- SAMPLE BUTTONS ----------------
st.markdown("### 🧪 Try Sample Messages")

spam_examples = [
    "Win ₹5000 cash now!!! Call immediately to claim your prize.",
    "Urgent! Your bank account will be blocked. Verify details now.",
    "Congratulations! You have won a free gift voucher."
]

safe_examples = [
    "Meeting postponed to 6 pm today. Please inform everyone.",
    "I will reach home by 8 pm. Don't wait for dinner.",
    "Please submit the assignment by tomorrow evening."
]


col1, col2 = st.columns(2)

with col1:
    if st.button("🚨 Spam Example"):
        st.session_state.message = random.choice(spam_examples)

with col2:
    if st.button("✅ Safe Example"):
        st.session_state.message = random.choice(safe_examples)


# ---------------- INPUT AREA ----------------
message = st.text_area(
    "📩 Enter your message to analyze:",
    value=st.session_state.get("message", ""),
    height=120
)

# ---------------- AI FUNCTIONS ----------------
def explain_message(message):
    vectorized = vectorizer.transform([message])
    feature_names = vectorizer.get_feature_names_out()
    scores = vectorized.toarray()[0]

    keywords = []
    for idx in scores.argsort()[-5:]:
        if scores[idx] > 0:
            keywords.append(feature_names[idx])

    return keywords

# ---------------- ANALYZE BUTTON ----------------
if st.button("🔎︎ Analyze"):

    if not message.strip():
        st.warning("⚠️ Please enter a message before clicking Analyze.")
    else:
        status = st.empty()

        with st.spinner("AI is thinking... please wait"):
            status.write("🔍 Analyzing message content...")
            time.sleep(1.5)

            status.write("🧠 Extracting NLP features...")
            time.sleep(1.5)

            status.write("📈 Evaluating spam risk...")
            time.sleep(1)

            # ---- ML PREDICTION ----
            vectorized = vectorizer.transform([message])
            prediction = model.predict(vectorized)[0]
            probabilities = model.predict_proba(vectorized)[0]

            spam_prob = probabilities[1] * 100
            ham_prob = probabilities[0] * 100

            keywords = explain_message(message)

            result = {
                "is_spam": prediction == 1,
                "spam_score": spam_prob,
                "ham_score": ham_prob,
                "keywords": keywords
            }

            status.empty()

        # ---------------- RESULT UI ----------------
        st.divider()

        if result["is_spam"]:
            st.error("### 🚨 Verdict: SPAM DETECTED")
        else:
            st.success("### ✅ Verdict: MESSAGE IS SAFE")

        m1, m2, m3 = st.columns(3)
        m1.metric("Spam Confidence", f"{result['spam_score']:.1f}%")
        m2.metric("Ham Confidence", f"{result['ham_score']:.1f}%")

        risk_val = (
            "HIGH" if result["spam_score"] > 80
            else "MEDIUM" if result["spam_score"] > 40
            else "LOW"
        )
        m3.metric("Risk Level", risk_val)

        st.progress(result["spam_score"] / 100)

        with st.expander("💡 See AI Explanation"):
            st.write(
                "The AI model analyzed the message and identified the following words "
                "as strong indicators that influenced its final decision:"
            )

            if result["keywords"]:
                st.write(" ".join([f"`{word}`" for word in result["keywords"]]))
            else:
                st.write("No dominant spam-related keywords were detected in this message.")

            st.info(
                "**How the AI works:** The system first converts the text into numerical "
                "features using Natural Language Processing (NLP). These features are then "
                "evaluated by a trained Machine Learning classifier, which calculates the "
                "probability of the message being spam based on learned patterns from past data."
            )



# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🚀 Built with Streamlit & Scikit-Learn | v2.0")
