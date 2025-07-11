
import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Make sure you saved this after training

st.title("💼 Employee Salary Prediction App")

st.markdown("Enter employee details to predict estimated salary")

# 13 Feature Inputs
age = st.number_input("Age", min_value=17, max_value=75, step=1)
workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Federal-gov', 'Self-emp-inc', 'Without-pay', 'Never-worked', 'NotListed'])
fnlwgt = st.number_input("Final Weight", min_value=10000, max_value=1000000, step=100)
marital_status = st.selectbox("Marital Status", ['Never-married', 'Married-civ-spouse', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent'])
occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces', 'Others'])
relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
gender = st.selectbox("Gender", ['Male', 'Female'])
capital_gain = st.number_input("Capital Gain", min_value=0, max_value=99999, step=100)
capital_loss = st.number_input("Capital Loss", min_value=0, max_value=99999, step=100)
hours_per_week = st.slider("Hours per Week", min_value=1, max_value=100)
native_country = st.selectbox("Native Country", ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Others'])
education_num = st.slider("Education Num (1-16)", min_value=1, max_value=16)

# Encoding maps (must match LabelEncoder from training)
workclass_map = {'Private': 4, 'Self-emp-not-inc': 6, 'Local-gov': 2, 'State-gov': 7,
                 'Federal-gov': 1, 'Self-emp-inc': 5, 'Without-pay': 8,
                 'Never-worked': 3, 'NotListed': 0}
marital_map = {'Never-married': 2, 'Married-civ-spouse': 1, 'Divorced': 0,
               'Separated': 4, 'Widowed': 5, 'Married-spouse-absent': 3}
occupation_map = {'Tech-support': 12, 'Craft-repair': 2, 'Other-service': 9, 'Sales': 10,
                  'Exec-managerial': 4, 'Prof-specialty': 8, 'Handlers-cleaners': 5,
                  'Machine-op-inspct': 6, 'Adm-clerical': 0, 'Farming-fishing': 3,
                  'Transport-moving': 13, 'Priv-house-serv': 7, 'Protective-serv': 11,
                  'Armed-Forces': 1, 'Others': 14}
relationship_map = {'Wife': 5, 'Own-child': 2, 'Husband': 1, 'Not-in-family': 3,
                    'Other-relative': 4, 'Unmarried': 0}
race_map = {'White': 4, 'Black': 1, 'Asian-Pac-Islander': 0,
            'Amer-Indian-Eskimo': 2, 'Other': 3}
gender_map = {'Male': 1, 'Female': 0}
native_map = {'United-States': 39, 'Mexico': 25, 'Philippines': 30, 'Germany': 11, 'Canada': 4, 'Others': 0}

# Apply encoding
workclass_enc = workclass_map[workclass]
marital_enc = marital_map[marital_status]
occupation_enc = occupation_map[occupation]
relationship_enc = relationship_map[relationship]
race_enc = race_map[race]
gender_enc = gender_map[gender]
native_enc = native_map[native_country]

# Prepare input
input_data = np.array([[age, workclass_enc, fnlwgt, marital_enc, occupation_enc,
                        relationship_enc, race_enc, gender_enc,
                        capital_gain, capital_loss, hours_per_week,
                        native_enc, education_num]])

# Scale
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Salary"):
    prediction = model.predict(input_scaled)
    st.success(f"💰 Estimated Income Group: {prediction[0]}")
