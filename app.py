import streamlit as st
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

APP_ROOT = os.path.dirname(__file__)
MODEL_PATH = os.path.join(APP_ROOT, "Loan_approval.pkl")
DATA_PATH = os.path.join(APP_ROOT, "loan_data.csv")

feature_cols = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 
                'cb_person_cred_hist_length', 'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT',
                'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE',
                'previous_loan_defaults_on_file_Yes']


def train_and_save_model():
    df = pd.read_csv(DATA_PATH)
    categorical_cols = df.select_dtypes(include='object').columns
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df[feature_cols]
    y = df['loan_status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)

    return clf


def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    if os.path.exists(DATA_PATH):
        return train_and_save_model()
    raise FileNotFoundError(f"Neither model ({MODEL_PATH}) nor data ({DATA_PATH}) were found.")


@st.cache_data
def get_model_cached():
    return load_model()


def main():
    st.title("Loan Approval Prediction")

    st.markdown("Provide applicant details and click Predict. The model will be trained automatically if `Loan_approval.pkl` is missing and `loan_data.csv` is present.")

    # Input fields
    person_age = st.number_input('person_age', value=30.0, format="%.2f")
    person_income = st.number_input('person_income', value=55000.0, format="%.2f")
    person_emp_exp = st.number_input('person_emp_exp', value=5.0, format="%.2f")
    loan_amnt = st.number_input('loan_amnt', value=20000.0, format="%.2f")
    loan_int_rate = st.number_input('loan_int_rate', value=12.0, format="%.2f")
    cb_person_cred_hist_length = st.number_input('cb_person_cred_hist_length', value=5.0, format="%.2f")

    col1, col2 = st.columns(2)
    with col1:
        loan_intent_PERSONAL = st.checkbox('loan_intent_PERSONAL', value=True)
        loan_intent_EDUCATION = st.checkbox('loan_intent_EDUCATION', value=False)
        loan_intent_HOMEIMPROVEMENT = st.checkbox('loan_intent_HOMEIMPROVEMENT', value=False)
    with col2:
        loan_intent_MEDICAL = st.checkbox('loan_intent_MEDICAL', value=False)
        loan_intent_VENTURE = st.checkbox('loan_intent_VENTURE', value=False)
        previous_loan_defaults_on_file_Yes = st.checkbox('previous_loan_defaults_on_file_Yes', value=False)

    input_df = pd.DataFrame([{ 
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
    }])
    if st.button('Predict'):
        try:
            model = get_model_cached()
            proba = model.predict_proba(input_df)[:, 1][0]
            pred = int(proba >= 0.5)
            st.success(f'Predicted probability of approval: {proba:.4f} ')
        except Exception as e:
            import streamlit as st
            import pandas as pd
            import os
            import io
            import joblib
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split

            # Paths
            APP_ROOT = os.path.dirname(__file__)
            MODEL_PATH = os.path.join(APP_ROOT, "Loan_approval.joblib")
            DATA_PATH = os.path.join(APP_ROOT, "loan_data.csv")

            feature_cols = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate',
                            'cb_person_cred_hist_length', 'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT',
                            'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE',
                            'previous_loan_defaults_on_file_Yes']


            def train_model_from_csv(csv_path):
                df = pd.read_csv(csv_path)
                # ensure features exist
                missing = [c for c in feature_cols + ['loan_status'] if c not in df.columns]
                if missing:
                    raise ValueError(f"Missing columns in CSV required for training: {missing}")

                # If there are object-type categorical columns, make dummies (but only if they match expected feature names)
                cat_cols = df.select_dtypes(include='object').columns.tolist()
                if cat_cols:
                    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

                # Keep only required feature columns (existing ones)
                X = df[feature_cols]
                y = df['loan_status']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                clf = RandomForestClassifier(random_state=42)
                clf.fit(X_train, y_train)
                return clf


            def load_model_from_disk(path):
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Model file not found at {path}")
                return joblib.load(path)


            def save_model_to_disk(model, path):
                joblib.dump(model, path)


            def main():
                st.title("Loan Approval Prediction")

                st.markdown("This app predicts loan approval probability. You may either upload a pre-trained model or train one from `loan_data.csv`.")

                st.sidebar.header("Model options")
                uploaded_model = st.sidebar.file_uploader("Upload trained model (.joblib or .pkl)", type=['joblib', 'pkl'])
                train_from_csv = st.sidebar.button("Train model from loan_data.csv")

                model = None
                # If user uploaded a model, try to load it
                if uploaded_model is not None:
                    try:
                        model_bytes = uploaded_model.read()
                        # joblib can load from a file-like object using BytesIO
                        model = joblib.load(io.BytesIO(model_bytes))
                        st.sidebar.success("Model loaded from upload.")
                    except Exception as e:
                        st.sidebar.error(f"Failed to load uploaded model: {e}")

                # If no uploaded model, try to load from disk
                if model is None and os.path.exists(MODEL_PATH):
                    try:
                        model = load_model_from_disk(MODEL_PATH)
                        st.sidebar.info("Loaded model from disk.")
                    except Exception as e:
                        st.sidebar.error(f"Failed to load model from disk: {e}")

                # Train model on demand
                if model is None and train_from_csv:
                    if os.path.exists(DATA_PATH):
                        try:
                            with st.spinner('Training model from CSV...'):
                                model = train_model_from_csv(DATA_PATH)
                                save_model_to_disk(model, MODEL_PATH)
                            st.sidebar.success('Training completed and model saved to disk.')
                        except Exception as e:
                            st.sidebar.error(f'Training failed: {e}')
                    else:
                        st.sidebar.error('loan_data.csv not found in project root; cannot train.')

                st.header('Applicant details')
                person_age = st.number_input('person_age', value=0.0, step=0.1)
                person_income = st.number_input('person_income', value=0.0, step=0.1)
                person_emp_exp = st.number_input('person_emp_exp', value=0.0, step=0.1)
                loan_amnt = st.number_input('loan_amnt', value=0.0, step=0.1)
                loan_int_rate = st.number_input('loan_int_rate', value=0.0, step=0.1)
                cb_person_cred_hist_length = st.number_input('cb_person_cred_hist_length', value=0.0, step=0.1)

                st.subheader('Loan intent flags')
                loan_intent_EDUCATION = st.checkbox('EDUCATION')
                loan_intent_HOMEIMPROVEMENT = st.checkbox('HOMEIMPROVEMENT')
                loan_intent_MEDICAL = st.checkbox('MEDICAL')
                loan_intent_PERSONAL = st.checkbox('PERSONAL')
                loan_intent_VENTURE = st.checkbox('VENTURE')
                previous_loan_defaults_on_file_Yes = st.checkbox('previous_loan_defaults_on_file_Yes')

                if st.button('Predict'):
                    if model is None:
                        st.error('No model available. Upload a model or train one from `loan_data.csv` in the sidebar.')
                    else:
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
