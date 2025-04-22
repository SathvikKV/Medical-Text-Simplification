import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Medical Text Simplification Evaluation", layout="wide")

# Load evaluation data
detailed_df = pd.read_csv("results/eval_detailed.csv")
summary_df = pd.read_csv("results/eval_summary.csv")

# ----------------------
# Header
# ----------------------
st.title("Medical Text Simplifier - Evaluation Dashboard")
st.markdown("""
This tool demonstrates the fine-tuning of a **T5-small model** to simplify complex medical content into plain English for patient understanding.

We compare the **baseline T5 model** against three fine-tuned configurations (`config_1`, `config_2`, `config_3`) on the **Cochrane Simplification test set**.  
Models are evaluated using **BLEU** and **ROUGE-L** scores.

---
""")

st.markdown("### 🔧 Model Configuration Comparison")

st.markdown("""
Below is a comparison of training configurations used for fine-tuning the T5-small model.  
We varied **learning rate**, **batch size**, and **epochs** across each config to observe their impact on simplification quality.

| Model      | Learning Rate | Batch Size | Epochs | Notes                          |
|------------|----------------|------------|--------|--------------------------------|
| `baseline` | —              | —          | —      | Pre-trained T5-small, no fine-tuning |
| `config_1` | `2e-5`         | `4`        | `3`    | Conservative learning rate and batch size |
| `config_2` | `3e-5`         | `8`        | `4`    | Larger batch, more aggressive training |
| `config_3` | `1e-4`         | `8`        | `2`    | Fastest training, highest LR |
""")


# ----------------------
# Section 1: Try Simplification on Your Own Text (Live Inference)
# ----------------------
from transformers import T5Tokenizer, T5ForConditionalGeneration

st.header("1. Try Simplification on Your Own Text")

user_input = st.text_area("Enter medical text:", value="The patient, a 67-year-old male with a history of hypertension, type 2 diabetes, and chronic kidney disease, presented with shortness of breath and chest tightness. An echocardiogram revealed reduced ejection fraction suggestive of systolic heart failure. BNP levels were elevated and chest X-ray confirmed pulmonary edema. Treatment included intravenous diuretics, initiation of ACE inhibitors, and oxygen supplementation. Cardiology recommended follow-up stress testing to evaluate for ischemic etiology.")

generate_btn = st.button("Simplify using all models")

if generate_btn and user_input.strip():
    st.subheader("Model Outputs (One Below the Other)")

    model_configs = {
    "baseline": "results/baseline_model",
    "config_1": "results/config_1",
    "config_2": "results/config_2",
    "config_3": "results/config_3"
}


    for model_name, model_path in model_configs.items():
        try:
            tokenizer = T5Tokenizer.from_pretrained(model_path)
            model = T5ForConditionalGeneration.from_pretrained(model_path)
        except Exception as e:
            st.error(f"Failed to load model '{model_name}' from {model_path}")
            st.exception(e)
            continue

        input_ids = tokenizer.encode("simplify: " + user_input, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.markdown(f"**🔹 {model_name}**")
        st.write(prediction)


st.markdown("---")

# ----------------------
# Section 2: Summary Results
# ----------------------
st.header("2. Summary of Evaluation Metrics")

st.markdown("Each model was evaluated using average BLEU and ROUGE-L scores over the test set.")

st.markdown("""
### 🧪 Evaluation Background & Methodology

To assess the performance of each model configuration, we evaluated them on the **Cochrane Simplification test set** using two key metrics:

- **BLEU Score (Bilingual Evaluation Understudy):**  
  Measures the degree of n-gram overlap between the model-generated simplification and the human-written reference summary. A higher score indicates closer alignment.

- **ROUGE-L Score (Recall-Oriented Understudy for Gisting Evaluation):**  
  Captures the longest common subsequence between the prediction and reference, rewarding fluency and structural similarity.

#### Methodology

1. **Test Dataset:**  
   A held-out set of expert-written medical simplifications from the Cochrane dataset.

2. **Generation:**  
   Each model generated predictions for the full test set (over 400 samples). 

3. **Scoring:**  
   Predictions were compared against the gold-standard reference using BLEU and ROUGE-L, averaged across all test samples.

4. **Comparison:**  
   Metrics were aggregated and visualized to identify the most effective fine-tuning strategy.

This evaluation provides a quantitative way to compare how well each fine-tuned model generalizes to unseen medical content, and how much it improves upon the baseline.
""")


# Display table
st.dataframe(summary_df.style.format({"bleu": "{:.4f}", "rouge_l": "{:.4f}"}), use_container_width=True)

# Bar charts
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
summary_df.plot(x="model", y="bleu", kind="bar", ax=ax[0], legend=False)
ax[0].set_title("Average BLEU Score")
ax[0].set_ylabel("Score")
ax[0].set_ylim(0, summary_df["bleu"].max() + 0.05)

summary_df.plot(x="model", y="rouge_l", kind="bar", ax=ax[1], legend=False, color='orange')
ax[1].set_title("Average ROUGE-L Score")
ax[1].set_ylabel("Score")
ax[1].set_ylim(0, summary_df["rouge_l"].max() + 0.05)

st.pyplot(fig)

# Best model
best_bleu = summary_df.sort_values("bleu", ascending=False).iloc[0]
st.success(f"✅ **Best configuration based on BLEU score:** `{best_bleu['model']}` with BLEU = {best_bleu['bleu']:.4f}")

st.markdown("---")

# ----------------------
# Section 3: Error Analysis
# ----------------------
st.header("3. Error Analysis – Low BLEU Examples")

st.markdown("""
Below are **5 examples** from the test set where model outputs had the **lowest BLEU scores**.

This helps uncover:
- Missing or incomplete simplifications
- Medical terms not simplified
- Over-simplification or hallucinated details
""")

# Filter and sort
worst_examples = detailed_df[(detailed_df["prediction"].notna())].sort_values("bleu").head(5)

for idx, row in worst_examples.iterrows():
    with st.expander(f"Example — Model: `{row['model']}` — BLEU: {row['bleu']:.4f}"):
        st.markdown("**Original Medical Text:**")
        st.write(row["input"])

        st.markdown("**Reference Simplification:**")
        st.info(row["reference"])

        st.markdown("**Model Prediction:**")
        st.warning(row["prediction"])



st.markdown("""
### 🔍 Error Analysis Methodology

In this section, we highlight **model failure points** by analyzing 5 examples from the **test set** where models scored the lowest BLEU scores.

#### Why Error Analysis?
Even if a model performs well on average, **low-scoring examples can reveal critical issues**:
- Is the model copying text verbatim?
- Is it hallucinating or removing important medical details?
- Is it oversimplifying at the cost of accuracy?

#### How We Selected the Examples
- We loaded the detailed evaluation results (input, prediction, reference, BLEU score).
- Sorted the dataset by BLEU score in ascending order.
- Displayed the bottom 5 examples from the lowest-performing model (`config_3` or `baseline`) where BLEU = 0.

#### What to Look For
Each example in this section displays:
- The **original medical abstract**
- The **human-written simplification** (reference)
- The **model's generated output**
- A flagged BLEU score = 0.0000

These edge cases often expose patterns such as:
- **Incomplete predictions** (e.g., stopping early)
- **Unchanged or poorly paraphrased jargon**
- **Loss of core facts or statistical outcomes**

#### Takeaway
While average metrics matter, qualitative review of poor predictions gives insight into where the model fails — and **how future improvements (e.g., better prompts, longer context windows, or more domain tuning)** could help.

---
""")



# ----------------------
# Footer
# ----------------------
st.markdown("📊 Built for the **Fine-Tuning a Large Language Model** assignment — demonstrating model training, evaluation, and analysis.")
st.caption("Author: Sathvik Vadavatha | Model: T5-small | Dataset: Cochrane Simplification (GEM)")
