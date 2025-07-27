# 💰 SalaryPreds

**SalaryPreds** is an AI-powered web application that predicts whether a person earns more than $50,000 per year based on U.S. Census data. Built with a focus on **explainability**, the app not only makes predictions but also **visualizes the top features driving the decision using SHAP**.

> Developed solo by [Kanif Kumbhar](https://github.com/Kanif-Kumbhar) as a demonstration of full-stack ML deployment with Streamlit and XGBoost.

---

## 🚀 Live Demo

Coming soon...  
(You can run it locally using the instructions below 👇)

---

## 🎯 Features

- ✅ Predict income class (`<=50K` or `>50K`) using a trained ML model
- ✅ End-to-end preprocessing with Scikit-learn Pipelines
- ✅ SHAP waterfall plot for individual prediction explanation
- ✅ SHAP force and summary plots for global interpretation
- ✅ Interactive Streamlit UI
- ✅ Lightweight and reproducible

---

## 📊 Model Evaluation

The model was trained and validated on the [UCI Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult). Several models were tested, and **XGBoost** was chosen for its superior performance.

### 📌 Confusion Matrix (XGBoost)

![Confusion Matrix](model_training/diagram/confusion_matrix_xgb.png)

### 📈 SHAP Force Plot

![SHAP Force Plot](model_training/diagram/shap_force.png)

### 📊 SHAP Summary Plot

![SHAP Summary Plot](model_training/diagram/shap_summary.png)

> All visualizations were generated using `matplotlib`, `seaborn`, and `shap`.

---

## 🧠 SHAP Explanation

To ensure transparency, every prediction is accompanied by a **SHAP waterfall plot** explaining how each input feature contributed to the outcome.

![SHAP Waterfall](model_training/diagram/shap_waterfall.png)

---

## 🛠 Tech Stack

| Layer           | Tools Used                              |
|------------------|------------------------------------------|
| 👨‍💻 ML Model      | XGBoost, Scikit-learn Pipelines         |
| 📊 Explainability | SHAP                                     |
| 🧪 Preprocessing  | OneHotEncoder, StandardScaler           |
| 🌐 Frontend      | Streamlit                               |
| 📈 Visualization | Matplotlib, Seaborn                      |
| 🐍 Packaging     | pipreqs, joblib                          |

---

## 💻 Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/TheOddDev/SalaryPreds.git
   cd SalaryPreds
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate        # On Windows
   source venv/bin/activate     # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## 📁 Folder Structure

```
SalaryPreds/
│
├── app.py                         # Streamlit frontend
├── model_training/
│   ├── model/                     # Trained model and preprocessor
│   │   ├── income_model.pkl
│   │   └── preprocessor.pkl
│   └── diagram/                   # Evaluation and SHAP plots
│       ├── confusion_matrix_xgb.png
│       ├── shap_force.png
│       ├── shap_summary.png
│       └── shap_waterfall.png
├── dataset/
│   └── adult.csv                  # Raw dataset
├── requirements.txt
└── README.md
```

---

## 🧑‍💻 Developer

Made with ❤️ by **[Kanif Kumbhar](https://github.com/Kanif-Kumbhar)**

> "I build transparent and ethical AI apps that anyone can understand."

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgments

- [UCI Machine Learning Repository – Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- [SHAP by Scott Lundberg](https://github.com/slundberg/shap)
- [Streamlit](https://streamlit.io/)

---