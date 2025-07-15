import joblib
import pandas as pd
import shap
import numpy as np
import json
import subprocess

# Load model
def load_model(path: str):
    return joblib.load(path)

# Preprocess input
def preprocess_input(input_data: dict) -> pd.DataFrame:
    df = pd.DataFrame([input_data])
    if "customer_id" in df.columns:
        df.drop(columns=["customer_id"], inplace=True)
    df["credit_mix"] = df["credit_mix"].map({"Good": 1, "Standard": 0, "Bad": -1})
    df["payment_behaviour"] = df["payment_behaviour"].map({
        "High_spent_Small_value_payments": 1,
        "Low_spent_Small_value_payments": 0,
        "High_spent_Large_value_payments": 2,
        "Low_spent_Large_value_payments": -1,
        "High_spent_Medium_value_payments": 3,
        "Low_spent_Medium_value_payments": -2
    })
    df["salary_range"] = df["salary_range"].map({
        "Very Low": 0, "Low": 1, "Medium": 2, "High": 3, "Very High": 4
    })

    return df

# Generate SHAP values
def get_shap_values(model, df):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(df)
    return shap_values

# Extract top contributing features
def get_top_features(df, shap_values, pred_index, top_n=5):
    shap_vals = shap_values.values[0, :, pred_index]  # shape (n_features,)
    feature_names = df.columns.tolist()
    top_indices = np.argsort(np.abs(shap_vals))[::-1][:top_n]
    return [(feature_names[i], shap_vals[i]) for i in top_indices]

# Save SHAP values
def save_shap_values(shap_values, path):
    shap_dict = shap_values.values[0].tolist()
    with open(path, "w") as f:
        json.dump(shap_dict, f)
    print(f"SHAP values saved to {path}")

def generate_prompt(prediction, top_features):
    feature_text = "\n".join(
        [f"- **{feat}**: contributed **{'+' if val >= 0 else '-'}{abs(val):.2f} points** toward the prediction." 
         for feat, val in top_features]
    )
    
    prompt = f"""
Category: {prediction}

Explanation:
1. The following features contributed most to the credit score prediction:
{feature_text}

2. Based on the above contributions, explain in natural language why this individual was assigned the category **{prediction}**.
- Make the reasoning clear and understandable.
- Focus on how positive contributions increased the score and negative contributions decreased it.
- Avoid technical jargon and explain like you are talking to a customer with no data science background.
"""
    return prompt

# Use Llama 3.2 via Ollama
def ask_ollama(prompt, model="llama3.2"):
    print("Querying Llama3 via Ollama...")
    response = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode(),
        stdout=subprocess.PIPE
    )
    output = response.stdout.decode()
    return output

# Main pipeline
def explain_credit_score(model_path, input_data):
    model = load_model(model_path)
    df = preprocess_input(input_data)
    prediction = model.predict(df)[0]

    shap_values = get_shap_values(model, df)

    # get predicted class index
    class_labels = model.classes_
    pred_index = list(class_labels).index(prediction)

    # pass pred_index to get_top_features
    top_features = get_top_features(df, shap_values, pred_index=pred_index)

    save_shap_values(shap_values, "shap_outputs/shap_values.json")
    prompt = generate_prompt(prediction, top_features)
    explanation = ask_ollama(prompt)

    print("\n--- PREDICTION ---")
    print(f"Predicted Category: {prediction}")
    print("\n--- LLM EXPLANATION ---")
    print(explanation)
    return prediction, explanation


# Example Usage
if __name__ == "__main__":
    with open("data/individual_input.json") as f:
        input_data = json.load(f)

    explain_credit_score("models/credit_score_model.pkl", input_data)
