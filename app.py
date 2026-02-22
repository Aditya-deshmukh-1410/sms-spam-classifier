import streamlit as st
import joblib
import numpy as np
import time
import random
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt

# ---------------- LOAD ML MODEL ----------------
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------- SAVE LOG FOR BI ----------------
def save_log(message, verdict, spam_conf, ham_conf, risk):
    file = "spam_logs.csv"
    row = {
        "message": message,
        "prediction": verdict,
        "spam_confidence": spam_conf,
        "ham_confidence": ham_conf,
        "risk_level": risk,
        "timestamp": datetime.now()
    }

    df = pd.DataFrame([row])

    if os.path.exists(file):
        df.to_csv(file, mode="a", header=False, index=False)
    else:
        df.to_csv(file, index=False)

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
    "FREE entry in 2 a weekly competition to win cash prizes.",
    "Congratulations! You have won a free gift voucher.",
    "You have won a lottery. Text CLAIM to receive your reward.",
    "URGENT! You have been selected for a cash reward. Reply YES now."
]

safe_examples = [
    "Meeting postponed to 6 pm today. Please inform everyone.",
    "I will reach home by 8 pm. Don't wait for dinner.",
    "Please submit the assignment by tomorrow evening.",
    "Hey, are we still meeting tomorrow evening?",
    "I will be late today, please have dinner without me."
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

def save_log(message, verdict, spam_conf, ham_conf, risk):
    file = "spam_logs.csv"
    row = {
        "message": message,
        "prediction": verdict,
        "spam_confidence": spam_conf,
        "ham_confidence": ham_conf,
        "risk_level": risk,
        "timestamp": datetime.now()
    }

    df = pd.DataFrame([row])

    if os.path.exists(file):
        df.to_csv(file, mode="a", header=False, index=False)
    else:
        df.to_csv(file, index=False)

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


            vectorized = vectorizer.transform([message])
            prediction = model.predict(vectorized)[0]
            probabilities = model.predict_proba(vectorized)[0]

            spam_prob = probabilities[1] * 100
            ham_prob = probabilities[0] * 100

            keywords = explain_message(message)

            risk_val = (
                "HIGH" if spam_prob > 80
                else "MEDIUM" if spam_prob > 40
                else "LOW"
            )

            verdict = "Spam" if prediction == 1 else "Ham"

            save_log(
                message,
                verdict,
                spam_prob,
                ham_prob,
                risk_val
            )

            

        # ---------------- RESULT UI ----------------
        st.divider()

        if verdict == "Spam":
            st.error("### 🚨 Verdict: SPAM DETECTED")
        else:
            st.success("### ✅ Verdict: MESSAGE IS SAFE")

        m1, m2, m3 = st.columns(3)
        m1.metric("Spam Confidence", f"{spam_prob:.1f}%")
        m2.metric("Ham Confidence", f"{ham_prob:.1f}%")
        m3.metric("Risk Level", risk_val)

        st.progress(spam_prob / 100)

        # ---------------- AI EXPLANATION ----------------
        with st.expander("💡 See AI Explanation"):
            st.write(
                "The AI model analyzed the message and identified the following words "
                "as strong indicators that influenced its final decision:"
            )

            if keywords:
                st.write(" ".join([f"`{word}`" for word in keywords]))
            else:
                st.write("No dominant spam-related keywords were detected.")

            st.info(
                "**How the AI works:** The system first converts the text into numerical "
                "features using Natural Language Processing (NLP). These features are then "
                "evaluated by a trained Machine Learning classifier."
            )

        # ---------------- BI DASHBOARD ----------------
        st.divider()
        st.header("🧠 Business Intelligence Insights")
        st.caption("Note: All charts represent a cumulative analysis of historical messages.")

        df = pd.read_csv("spam_logs.csv")

        total = len(df)
        spam_count = len(df[df["prediction"] == "Spam"])
        ham_count = len(df[df["prediction"] == "Ham"])
        high_risk = len(df[df["risk_level"] == "HIGH"])

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Messages", total)
        k2.metric("Spam", spam_count)
        k3.metric("Safe", ham_count)
        k4.metric("High Risk", high_risk)

        st.subheader("📊 Pie Chart Distribution")

        labels = ["Spam", "Ham"]
        sizes = [spam_count, ham_count]
        colors = ["#E74C3C", "#2ECC71"]  # Red, Green

        fig1, ax1 = plt.subplots()
        ax1.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5}
        )
        ax1.axis("equal")
        st.pyplot(fig1)

        st.subheader("⚠️ Risk Level Distribution")

        risk_counts = df["risk_level"].value_counts().reindex(
            ["LOW", "MEDIUM", "HIGH"], fill_value=0
        )

        colors = ["#2ECC71", "#F39C12", "#E74C3C"]

        fig2, ax2 = plt.subplots()
        ax2.bar(
            risk_counts.index,
            risk_counts.values,
            color=colors
        )

        ax2.set_xlabel("Risk Level")
        ax2.set_ylabel("Number of Messages")
        ax2.set_title("Message Risk Distribution")

        st.pyplot(fig2)

        st.subheader("📈 Spam vs Ham Comparison")

        spam_cum = df["prediction"].eq("Spam").cumsum()
        ham_cum = df["prediction"].eq("Ham").cumsum()

        fig4, ax4 = plt.subplots()
        ax4.plot(spam_cum, label="Spam Messages")
        ax4.plot(ham_cum, label="Ham Messages")

        ax4.set_xlabel("Total Messages Analyzed")
        ax4.set_ylabel("Count")
        ax4.set_title("Spam vs Ham Trend")
        ax4.legend()

        st.pyplot(fig4)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🚀 Built with Streamlit & Scikit-Learn | v2.0")