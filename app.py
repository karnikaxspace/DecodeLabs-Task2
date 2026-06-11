"""
Iris Classifier – AI Project 2 (DecodeLabs)
Supervised Learning with K-Nearest Neighbors
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Page config ----------
st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="centered")

st.title("🌸 Iris Flower Classifier")
st.markdown("### *Supervised Learning – K‑Nearest Neighbors*")

st.markdown("""
Enter the measurements of an iris flower. The AI model (trained on 150 labelled examples) will predict its species.
""")

# ---------- Load & prepare data (cached) ----------
@st.cache_data
def load_and_train():
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardise features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train KNN (K=5)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return knn, scaler, feature_names, target_names, X_test_scaled, y_test, y_pred, accuracy

knn, scaler, feature_names, target_names, X_test_scaled, y_test, y_pred, accuracy = load_and_train()

# ---------- Sidebar: model performance ----------
st.sidebar.header("📊 Model Performance (on test set)")
st.sidebar.metric("Accuracy", f"{accuracy:.2%}")
if st.sidebar.checkbox("Show Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.sidebar.pyplot(fig)

if st.sidebar.checkbox("Show Classification Report"):
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    st.sidebar.dataframe(pd.DataFrame(report).transpose())

# ---------- User input sliders ----------
st.subheader("📏 Enter flower measurements (cm)")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.5, 0.1)
    sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0, 0.1)
with col2:
    petal_length = st.slider("Petal Length", 1.0, 7.0, 3.5, 0.1)
    petal_width = st.slider("Petal Width", 0.1, 2.5, 1.2, 0.1)

# ---------- Prediction ----------
if st.button("🔍 Classify Iris Species", type="primary"):
    # Create array for the input
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_features)
    
    # Predict
    prediction = knn.predict(input_scaled)[0]
    probabilities = knn.predict_proba(input_scaled)[0]
    
    species = target_names[prediction]
    confidence = probabilities[prediction]
    
    # Display result (fixed contrast)
    st.subheader("🌼 Prediction Result")
    
    st.markdown(f"""
    <div style="background-color: #2e7d64; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #1b5a48;">
        <h2 style="color: white; margin: 0;">{species}</h2>
        <p style="color: #e0f2f1; font-size: 16px; margin-top: 10px;">Confidence: <strong>{confidence:.1%}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show probability distribution
    st.markdown("#### Confidence per species")
    prob_df = pd.DataFrame({
        "Species": target_names,
        "Probability": probabilities
    })
    st.bar_chart(prob_df.set_index("Species"))
    
    st.caption(f"*Model accuracy on test set: {accuracy:.2%}*")

# ---------- Footer ----------
st.markdown("---")
st.caption("🧠 Trained on Iris dataset (150 samples) | KNN (k=5) | Features standardised")