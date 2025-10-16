# Loan Approval Prediction (Streamlit)

#APP LINK :-https://loanapproval-f4rmprozfpxjdj52yrmglu.streamlit.app/

This repository contains a small Streamlit app that predicts loan approval probability using a Random Forest model. The app was created by extracting the code from `loan_approval.ipynb` and wiring it to a web UI.

Contents
- `app.py` - Streamlit application. Loads `Loan_approval.pkl` if present, otherwise trains a RandomForest from `loan_data.csv` and saves the pickle.
- `loan_data.csv` - (original dataset used in the notebook). Keep it alongside the app if you want the app to be able to train the model automatically.
- `Loan_approval.pkl` - optional pre-trained model (if you have it, place it next to `app.py` so the app loads it immediately).
- `requirements.txt` - dependencies.

Quick start (Windows PowerShell)

1. Open PowerShell in the project folder (where `app.py` lives).
2. Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\requirements.txt
```

3. Run the Streamlit app:

```powershell
streamlit run .\app.py
```

The Streamlit UI will open in your browser. If it doesn't open automatically, look for the local URL in the terminal output (usually http://localhost:8501).

How the app works
- On startup the app attempts to load `Loan_approval.pkl` from the project folder.
- If the pickle is missing and `loan_data.csv` is available, the app will train a RandomForest model using the same features from the notebook and save `Loan_approval.pkl` for later runs.
- The UI lets you enter applicant numeric values and select loan intent flags. Clicking Predict returns the predicted probability and a binary label (1 = approve when probability >= 0.5).

Expected input fields (names used by the model)
- person_age: number
- person_income: number
- person_emp_exp: number
- loan_amnt: number
- loan_int_rate: number
- cb_person_cred_hist_length: number
- loan_intent_EDUCATION: checkbox (0/1)
- loan_intent_HOMEIMPROVEMENT: checkbox (0/1)
- loan_intent_MEDICAL: checkbox (0/1)
- loan_intent_PERSONAL: checkbox (0/1)
- loan_intent_VENTURE: checkbox (0/1)
- previous_loan_defaults_on_file_Yes: checkbox (0/1)

Troubleshooting
- If Streamlit reports "module not found" for streamlit/pandas/sklearn, ensure you installed dependencies in the same environment you're running Streamlit from. Use `pip show streamlit` to verify.
- If the app raises a FileNotFoundError about `loan_data.csv` and you don't have `Loan_approval.pkl`, either place `loan_data.csv` in the project folder or add `Loan_approval.pkl` there.
- If the app opens but Predict throws errors, check the terminal output; it will show the Python traceback. Common causes: mismatched feature columns (if `loan_data.csv` schema changed) or missing columns in a provided pickle.

Next improvements (optional)
- Serialize and store preprocessing (ColumnTransformer / OneHotEncoder + model) together so saving/loading is robust.
- Add an "Upload model" button to the UI so you can upload `Loan_approval.pkl` from the browser.
- Add input validation and friendly error messages for empty/missing numeric fields.

If you want, I can add an upload button for a pickle model and input validation now — tell me which you prefer.

Conclusion
This project provides a simple, reproducible pipeline for predicting loan approval using a Random Forest model and a Streamlit UI for quick experimentation. It’s intended as a lightweight demo — you can run it locally, inspect the code, retrain the model from loan_data.csv, or drop in a pre-trained Loan_approval.pkl.

If you plan to take this further, consider serializing the preprocessing with the model, adding input validation, and hardening the app for production (authentication, logging, and tests). Contributions, bug reports, or questions are welcome — open an issue or send a PR with improvements.
