import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Income Prediction App",
    page_icon="💰",
    layout="wide"
)

# --- Load Model and Preprocessor ---
@st.cache_resource
def load_artifacts():
    model = joblib.load('../model_training/models/xgb_model.joblib')
    preprocessor = joblib.load('../model_training/models/preprocessor.joblib')
    return model, preprocessor

model, preprocessor = load_artifacts()

# --- SHAP Explainer ---
@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)  # TreeExplainer for XGBoost

explainer = load_explainer(model)

# --- Load Dataset for UI Options ---
@st.cache_data
def load_original_data():
    return pd.read_csv('../dataset/adult.csv')

df_original = load_original_data()

# --- App Title and Description ---
st.title("Income Bracket Prediction 💸")
st.write(
    "This app predicts whether an individual's income is more or less than $50K per year. "
    "Input your details in the sidebar and see the prediction along with an explanation of the result."
)

# --- Sidebar for User Input ---
st.sidebar.header("User Input Features")

def get_user_input(df_original):
    age = st.sidebar.slider('Age', 17, 90, 35)
    workclass = st.sidebar.selectbox('Work Class', df_original['workclass'].unique())
    fnlwgt = st.sidebar.number_input('Final Weight (fnlwgt)', 12285, 1490400, 178356)
    education = st.sidebar.selectbox('Education', df_original['education'].unique())
    educational_num = st.sidebar.slider('Years of Education', 1, 16, 10)
    marital_status = st.sidebar.selectbox('Marital Status', df_original['marital-status'].unique())
    occupation = st.sidebar.selectbox('Occupation', df_original['occupation'].unique())
    relationship = st.sidebar.selectbox('Relationship', df_original['relationship'].unique())
    race = st.sidebar.selectbox('Race', df_original['race'].unique())
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    capital_gain = st.sidebar.number_input('Capital Gain', 0, 99999, 0)
    capital_loss = st.sidebar.number_input('Capital Loss', 0, 4356, 0)
    hours_per_week = st.sidebar.slider('Hours per Week', 1, 99, 40)
    native_country = st.sidebar.selectbox('Native Country', df_original['native-country'].unique())

    input_dict = {
        'age': age, 'workclass': workclass, 'fnlwgt': fnlwgt, 'education': education,
        'educational-num': educational_num, 'marital-status': marital_status,
        'occupation': occupation, 'relationship': relationship, 'race': race,
        'gender': gender, 'capital-gain': capital_gain, 'capital-loss': capital_loss,
        'hours-per-week': hours_per_week, 'native-country': native_country
    }
    return pd.DataFrame([input_dict])

user_input_df = get_user_input(df_original)

# --- Debug: Display current input ---
st.subheader("Your Input Summary")
st.write(user_input_df)

# --- Prediction and Explanation ---
st.header("Prediction Result")

# Transform input
transformed_input = preprocessor.transform(user_input_df)

# Make prediction
prediction_proba = model.predict_proba(transformed_input)
prediction = model.predict(transformed_input)

# Display prediction
if prediction[0] == 1:
    st.success("🎯 **Predicted Income: > $50K**")
else:
    st.error("💼 **Predicted Income: <= $50K**")

st.write(f"🔍 Confidence: **{prediction_proba[0][prediction[0]]*100:.2f}%**")

# --- SHAP Explanation Plot ---
st.header("Reason for Prediction")
st.write(
    "The plot below explains how each feature contributed to the final prediction. "
    "**Red bars** push the prediction higher (towards '>50K'), while **blue bars** push it lower."
)

# SHAP Plot
def shap_waterfall_plot(input_data):
    transformed = preprocessor.transform(input_data)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    shap_values = explainer(transformed)

    # Get correct feature names
    numeric_features = preprocessor.named_transformers_['num'].get_feature_names_out()
    categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
    shap_values.feature_names = list(numeric_features) + list(categorical_features)

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12
    })

    fig, ax = plt.subplots(figsize=(14, 6))
    shap.plots.waterfall(shap_values[0], max_display=15, show=False)
    plt.title("Top Features Contributing to Income Prediction", fontsize=16, weight='bold')
    plt.tight_layout()
    return fig

# Display SHAP plot
with st.spinner('Generating explanation...'):
    fig = shap_waterfall_plot(user_input_df)
    st.pyplot(fig, use_container_width=True)

with st.expander("ℹ️ What do E(x) and f(x) mean?"):
    st.markdown("""
    - **E(x)**: This is the model’s **baseline prediction** — what the model expects if it doesn’t know anything about the individual input.
    - **f(x)**: This is the **actual prediction** for your input values.
    - **SHAP values** show how much each feature moved the prediction from **E(x)** to **f(x)**.

    **Color meanings in the SHAP plot:**
    - 🔴 **Red bars**: Push the prediction **higher** (toward `> $50K`)
    - 🔵 **Blue bars**: Push the prediction **lower** (toward `<= $50K`)
    """)
    
st.markdown(
    """
    <hr style="margin-top: 50px;">
    <div style="text-align: center; padding: 10px; font-size: 14px; color: gray;">
        Code. Coffee. Creation. — <strong>Kanif Kumbhar</strong>
    </div>
    """,
    unsafe_allow_html=True
)