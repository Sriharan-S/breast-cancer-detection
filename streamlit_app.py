import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('BreastCancer.csv')
    return df

# Preprocess data
def preprocess_data(df):
    # Handle categorical columns
    label_encoder = LabelEncoder()
    df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])
    return df

# Train and evaluate model
def train_svm(df):
    X = df[['radius_mean', 'texture_mean', 'perimeter_mean',
            'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
            'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
            'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
            'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
            'fractal_dimension_se', 'radius_worst', 'texture_worst',
            'perimeter_worst', 'area_worst', 'smoothness_worst',
            'compactness_worst', 'concavity_worst', 'concave points_worst',
            'symmetry_worst', 'fractal_dimension_worst']]
    y = df['diagnosis']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the SVM classifier
    svm_clf = SVC(random_state=42)
    svm_clf.fit(X_train_scaled, y_train)

    return svm_clf, scaler, X.columns

# Make predictions based on user input
def predict(svm_clf, scaler, feature_input):
    input_df = pd.DataFrame([feature_input], columns=feature_input.keys())
    input_scaled = scaler.transform(input_df)
    prediction = svm_clf.predict(input_scaled)
    return "Malignant" if prediction[0] == 1 else "Benign"

# Main app layout
def main():
    st.set_page_config(page_title="SVM Breast Cancer Prediction", layout="wide")
    st.title("🏥 Breast Cancer Prediction using SVM")

    df = load_data()
    df = preprocess_data(df)
    svm_clf, scaler, feature_names = train_svm(df)

    # User input section
    st.write("### 🔢 Enter the Feature Values for Prediction")

    user_input = {}
    for feature in feature_names:
        user_input[feature] = st.number_input(f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))

    # Predict button
    if st.button("🔍 Predict"):
        prediction = predict(svm_clf, scaler, user_input)
        st.write(f"### 🩺 Prediction: **{prediction}**")

if __name__ == "__main__":
    main()
