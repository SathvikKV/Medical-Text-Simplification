import os
import json
import nltk
import pandas as pd
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Download NLTK punkt tokenizer
nltk.download("punkt")

# Load test data
with open("data/test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# Define model paths
configs = {
    "baseline": "t5-small",
    "config_1": "results/config_1",
    "config_2": "results/config_2",
    "config_3": "results/config_3"
}

# Evaluation function
def evaluate_model(model_path, tokenizer_path=None):
    tokenizer_path = tokenizer_path or model_path
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.eval()

    smooth = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    results = []
    for item in test_data[:100]:  # Evaluate on first 100 items for speed
        input_text = "simplify: " + item["source"]
        input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
        prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        reference = item["target"]

        ref_tokens = nltk.word_tokenize(reference.lower())
        pred_tokens = nltk.word_tokenize(prediction.lower())
        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
        rouge_l = scorer.score(reference, prediction)["rougeL"].fmeasure

        results.append({
            "input": item["source"],
            "reference": reference,
            "prediction": prediction,
            "bleu": bleu,
            "rouge_l": rouge_l
        })
    return results

# Run evaluations for all models
all_results = []
for model_label, path in configs.items():
    print(f"Evaluating: {model_label}")
    model_scores = evaluate_model(path)
    for entry in model_scores:
        entry["model"] = model_label
    all_results.extend(model_scores)

# Save full evaluation results
df_results = pd.DataFrame(all_results)
os.makedirs("results", exist_ok=True)
df_results.to_csv("results/eval_detailed.csv", index=False)

# Compute and save average scores
df_summary = df_results.groupby("model")[["bleu", "rouge_l"]].mean().reset_index()
df_summary.to_csv("results/eval_summary.csv", index=False)

print("\n✅ Evaluation complete.")
print("📊 Saved full results to: results/eval_detailed.csv")
print("📈 Saved average scores to: results/eval_summary.csv")
