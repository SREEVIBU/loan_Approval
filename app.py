import streamlit as st
import pandas as pd
import os
import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

APP_ROOT = os.path.dirname(__file__)
MODEL_JOBLIB = os.path.join(APP_ROOT, "Loan_approval.joblib")
MODEL_PKL = os.path.join(APP_ROOT, "Loan_approval.pkl")
DATA_PATH = os.path.join(APP_ROOT, "loan_data.csv")

feature_cols = [
    'person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate',
    'cb_person_cred_hist_length', 'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT',
    'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE',
    'previous_loan_defaults_on_file_Yes'
]


def load_model_from_disk():
    # Try joblib first
    if os.path.exists(MODEL_JOBLIB):
        try:
            return joblib.load(MODEL_JOBLIB)
        except Exception:
            pass
    # Try pickle
    if os.path.exists(MODEL_PKL):
        try:
            with open(MODEL_PKL, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    return None


def train_and_save_from_csv():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError('loan_data.csv not found in project folder; cannot train model.')
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in feature_cols + ['loan_status'] if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns in CSV required for training: {missing}')

    # If there are object columns that were one-hot encoded in the notebook, attempt to convert them.
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df[feature_cols]
    y = df['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Save to joblib for reliability
    joblib.dump(clf, MODEL_JOBLIB)
    return clf


def get_or_train_model():
    model = load_model_from_disk()
    if model is not None:
        return model
    # try training if CSV exists
    if os.path.exists(DATA_PATH):
        return train_and_save_from_csv()
    return None


def main():
    st.title('Loan Approval Predictor')

    st.write('This app will automatically load a model from disk (Loan_approval.joblib or Loan_approval.pkl).')
    st.write('If no model is found and `loan_data.csv` is present, it will train a model automatically.')

    with st.spinner('Loading model...'):
        try:
            model = get_or_train_model()
        except Exception as e:
            st.error(f'Model load/train failed: {e}')
            model = None

    if model is None:
        st.error('No model is available. Add `Loan_approval.joblib` or `Loan_approval.pkl` to the project folder or provide `loan_data.csv` to allow training.')

    st.header('Applicant details')
    person_age = st.number_input('person_age', value=0.0)
    person_income = st.number_input('person_income', value=0.0)
    person_emp_exp = st.number_input('person_emp_exp', value=0.0)
    loan_amnt = st.number_input('loan_amnt', value=0.0)
    loan_int_rate = st.number_input('loan_int_rate', value=0.0)
    cb_person_cred_hist_length = st.number_input('cb_person_cred_hist_length', value=0.0)

    st.subheader('Loan intent flags')
    loan_intent_EDUCATION = st.checkbox('EDUCATION')
    loan_intent_HOMEIMPROVEMENT = st.checkbox('HOMEIMPROVEMENT')
    loan_intent_MEDICAL = st.checkbox('MEDICAL')
    loan_intent_PERSONAL = st.checkbox('PERSONAL')
    loan_intent_VENTURE = st.checkbox('VENTURE')
    previous_loan_defaults_on_file_Yes = st.checkbox('previous_loan_defaults_on_file_Yes')

    if st.button('Predict'):
        if model is None:
            st.error('No model available. Place a model file or loan_data.csv in the project folder and reload.')
            return

        data = {
            'person_age': float(person_age),
            'person_income': float(person_income),
            'person_emp_exp': float(person_emp_exp),
            'loan_amnt': float(loan_amnt),
            'loan_int_rate': float(loan_int_rate),
            'cb_person_cred_hist_length': float(cb_person_cred_hist_length),
            'loan_intent_EDUCATION': int(loan_intent_EDUCATION),
            'loan_intent_HOMEIMPROVEMENT': int(loan_intent_HOMEIMPROVEMENT),
            'loan_intent_MEDICAL': int(loan_intent_MEDICAL),
            'loan_intent_PERSONAL': int(loan_intent_PERSONAL),
            'loan_intent_VENTURE': int(loan_intent_VENTURE),
            'previous_loan_defaults_on_file_Yes': int(previous_loan_defaults_on_file_Yes)
        }

        X_new = pd.DataFrame([data], columns=feature_cols)
        try:
            proba = model.predict_proba(X_new)[:, 1][0]
            pred = int(proba >= 0.5)
            st.success(f'Predicted probability of approval: {proba:.4f}')
            st.info(f'Predicted label: {pred}')
        except Exception as e:
            st.error(f'Prediction failed: {e}')


if __name__ == '__main__':
    main()
