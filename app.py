import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import streamlit as st

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

@st.cache_resource
def train_model(X_train, y_train):
    model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    return model

def main():
    st.title("🤖 Intelligent Chatbot Assistant")
    st.write("Type your message below and get the intent detected.")

    uploaded_file = st.file_uploader("Upload your balanced dataset CSV file", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)

        # Prepare data
        X = df['utterance']
        y = df['intent']

        # Check class distribution
        class_counts = y.value_counts()
        st.write("Class distribution:\n", class_counts)

        # Remove classes with less than 2 instances
        classes_to_keep = class_counts[class_counts >= 2].index
        df_filtered = df[df['intent'].isin(classes_to_keep)]

        # Prepare data again after filtering
        X_filtered = df_filtered['utterance']
        y_filtered = df_filtered['intent']

        # Split data
        if len(y_filtered.value_counts()) < 2:
            st.warning("Not enough classes to perform stratified split.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered, stratify=y_filtered, test_size=0.2, random_state=42
            )

            model = train_model(X_train, y_train)

            user_input = st.text_input("🗣️ You:")
            if user_input:
                prediction = model.predict([user_input])[0]
                st.write(f"🤖 Intent Detected: **{prediction}**")

            if st.button("Show Classification Report"):
                y_pred = model.predict(X_test)
                report = classification_report(y_test, y_pred)
                st.text("📊 Classification Report:\n" + report)
    else:
        st.info("Please upload the 'Balanced_Chatbot_Intent_Dataset1.csv' file to continue.")

if __name__ == "__main__":
    main()
