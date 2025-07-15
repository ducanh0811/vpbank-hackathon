import json
import os
from LLM_Explaination import explain_credit_score


def load_unstructured_data(customer_id, path="outputs/unstructured_data"):
    try:
        with open(f"{path}/{customer_id}.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"notes": "No unstructured data available."}


def combine_prompt(structured: dict, unstructured: dict):
    features = "\n".join([f"- {k}: {v:.2f}" for k, v in structured["top_features"]])
    return f"""
This user has a predicted credit score category of **{structured["prediction"]}**.

Top SHAP features:
{features}

Additional notes from behavior and search history:
{unstructured.get("notes", "N/A")}

Now explain why this user received this credit score in a clear, simple, customer-friendly way using both structured and behavioral data.
"""


def final_explanation(model_path, input_json, final_output_path):
    with open(input_json, "r", encoding="utf-8") as f:
        input_data = json.load(f)
    customer_id = input_data.get("customer_id", "unknown")

    # Run structured explanation first
    explain_credit_score(
        model_path=model_path,
        input_data=input_data,
        output_dir="outputs/structured_data"
    )

    # Load results
    with open(f"outputs/structured_data/{customer_id}.json", "r", encoding="utf-8") as f:
        structured = json.load(f)

    unstructured = load_unstructured_data(customer_id)

    # Final LLM prompt
    final_prompt = combine_prompt(structured, unstructured)
    explanation = explain_credit_score(model_path, input_data, output_dir="outputs/final")[1]

    # Save final result
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump({
            "customer_id": customer_id,
            "final_explanation": explanation
        }, f, indent=2, ensure_ascii=False)

    print(f"✅ Full explanation saved at: {final_output_path}")


if __name__ == "__main__":
    final_explanation(
        model_path="models/credit_score_model.pkl",
        input_json="data/individual.json",
        final_output_path="final_explain/final_result.json"
    )
