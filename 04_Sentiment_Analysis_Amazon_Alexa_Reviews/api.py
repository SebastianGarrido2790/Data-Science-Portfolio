from flask import Flask, request, render_template_string
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = Flask(__name__)

MODEL_PATH = "C:\Users\Desktop\Models\distilbert_binary_model"

# Load model and tokenizer
try:
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
except Exception as e:
    raise Exception(f"Failed to load model: {str(e)}")


# Prediction function
def predict_sentiment(review: str) -> dict:
    inputs = tokenizer(
        review,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_prob = probs[0][1].item()
        pred_label = 1 if pred_prob >= 0.5 else 0

    return {
        "sentiment": "positive" if pred_label == 1 else "negative",
        "confidence": pred_prob if pred_label == 1 else 1 - pred_prob,
    }


# Serve index.html and handle prediction
with open("static/index.html", "r") as f:
    index_html = f.read()


@app.route("/", methods=["GET"])
def index():
    return render_template_string(index_html, result=None, error=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        review = request.form.get("review")
        if not review:
            return render_template_string(
                index_html, result=None, error="Review is required"
            )
        result = predict_sentiment(review)
        return render_template_string(index_html, result=result, error=None)
    except Exception as e:
        return render_template_string(index_html, result=None, error=str(e))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
