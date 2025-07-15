import json
from main import explain_credit_score  # sửa theo tên file thật của bạn

def save_explanation_to_json(model_path: str, input_path: str, output_path: str):
    # Load input data từ file
    with open(input_path, "r") as f:
        input_data = json.load(f)
    
    user_id = input_data.get("customer_id", None)  # Lấy ID nếu có

    # Gọi pipeline chính
    prediction, explanation = explain_credit_score(model_path, input_data)

    # Tạo dictionary để lưu
    output = {
        "customer_id": user_id,
        "prediction": prediction,
        "explanation": explanation.strip()
    }

    # Lưu vào JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ Đã lưu giải thích vào: {output_path}")


# Ví dụ chạy
if __name__ == "__main__":
    save_explanation_to_json(
        model_path="models/credit_score_model.pkl",
        input_path="data/individual_input.json",
        output_path="outputs/structured_data/llm_explanation.json"
    )


