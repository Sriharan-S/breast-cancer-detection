# 🏥 SVM Breast Cancer Prediction App

This repository contains a Streamlit web application that predicts breast cancer diagnosis (Benign or Malignant) using a Support Vector Machine (SVM) classifier. The app allows users to input various features related to breast cancer and get a prediction result.

## 🚀 Features

- **Support Vector Machine (SVM)** classification.
- **User-friendly Streamlit interface** for entering feature values.
- **Preprocessing pipeline** including scaling and label encoding.
- **Interactive prediction** with real-time input adjustments.

## 📊 Dataset

The app uses the **Breast Cancer Wisconsin Dataset**. Make sure you have a CSV file named `BreastCancer.csv` in the project directory. The dataset should include diagnostic data with various numerical features.

## 🛠️ Requirements

To run this app, install the following dependencies:

```bash
pip install streamlit pandas scikit-learn matplotlib seaborn
```

## ⚙️ Setup and Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/breast-cancer-svm.git
   cd breast-cancer-svm
   ```

2. Make sure the dataset (`BreastCancer.csv`) is in the root directory.

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

4. Open the app in your browser at:

   ```
   http://localhost:8501
   ```

## 📝 How to Use the App

1. **Feature Input Section:** Adjust the feature values using the input fields provided.
2. **Predict Button:** Click the "🔍 Predict" button to generate a diagnosis.
3. **Prediction Result:** The app will display whether the diagnosis is "Benign" or "Malignant."

## 📂 File Structure

```
.
├── app.py              # Main Streamlit application
├── BreastCancer.csv    # Dataset file (to be provided)
└── README.md          # This documentation
```

## 🎯 Model Details

- **Model:** Support Vector Machine (SVM)
- **Preprocessing:** Standard scaling, label encoding
- **Metrics:** Accuracy, precision, recall, F1 score

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License.

---

🌟 **Author:** [Sriharan S](https://github.com/Sriharan-S)
