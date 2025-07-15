import json
import os
from LLM_Explaination import explain_credit_score  # pipeline hiện tại
from pathlib import Path


def load_unstructured_data(customer_id: str, unstructured_folder: str = "unstructured_data"):
    path = os.path.join(unstructured_folder, f"{customer_id}.json")
    if not os.path.exists(path):
        print(f"⚠️ Không tìm thấy unstructured data cho ID {customer_id}")
        return None
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def enhance_prompt_with_unstructured(prompt: str, unstructured_data: dict) -> str:
    if not unstructured_data:
        return prompt

    search_queries = unstructured_data.get("google_search_history", [])
    note = unstructured_data.get("notes", "")

    formatted_queries = "\n".join(f"- {q}" for q in search_queries)
    
    enhancement = f"""
3. The user has the following recent Google search queries, which may reflect their current concerns or interests:
{formatted_queries}

4. Additional Notes:
{note}

Based on both the model's explanation and these behavioral cues, provide a more personalized reasoning behind the credit category.
"""
    return prompt.strip() + "\n\n" + enhancement.strip()


def explain_with_unstructured(model_path: str, input_path: str, unstructured_folder: str, output_path: str):
    # Load input JSON (structured)
    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    customer_id = input_data.get("customer_id")
    if not customer_id:
        raise ValueError("Missing 'customer_id' in structured input.")

    # Run structured SHAP + LLM explain
    from LLM_Explaination import generate_prompt, get_shap_values, preprocess_input, load_model, get_top_features

    model = load_model(model_path)
    df = preprocess_input(input_data)
    prediction = model.predict(df)[0]
    shap_values = get_shap_values(model, df)
    class_labels = model.classes_
    pred_index = list(class_labels).index(prediction)
    top_features = get_top_features(df, shap_values, pred_index=pred_index)

    structured_prompt = generate_prompt(prediction, top_features)

    # Load unstructured data
    unstructured_data = load_unstructured_data(customer_id, unstructured_folder)

    # Merge into full prompt
    final_prompt = enhance_prompt_with_unstructured(structured_prompt, unstructured_data)

    # Run Ollama
    from LLM_Explaination import ask_ollama
    explanation = ask_ollama(final_prompt)

    # Save final explanation
    output = {
        "customer_id": customer_id,
        "prediction": prediction,
        "explanation": explanation.strip()
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ Đã lưu giải thích cuối cùng vào: {output_path}")


# Example run
if __name__ == "__main__":
    explain_with_unstructured(
        model_path="models/credit_score_model.pkl",
        input_path="data/individual_input.json",
        unstructured_folder="unstructured_data",
        output_path="outputs/full_explanation.json"
    )
