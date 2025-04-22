import os
import json
import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset

def load_local_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)

def preprocess_function(examples, tokenizer, max_input_len=512, max_target_len=128):
    inputs = ["simplify: " + text for text in examples["source"]]
    model_inputs = tokenizer(inputs, max_length=max_input_len, truncation=True, padding="max_length")
    labels = tokenizer(
        examples["target"],
        max_length=max_target_len,
        truncation=True,
        padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_model(train_dataset, val_dataset, tokenizer, config_id, learning_rate, batch_size, num_epochs):
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    output_dir = f"./results/config_{config_id}"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        do_train=True,
        do_eval=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
    )

    print(f"Training config {config_id}...")
    trainer.train()

    print(f"Saving model for config {config_id}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def save_baseline_model():
    print("Saving baseline (untrained) T5-small model...")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    os.makedirs("./results/baseline_model", exist_ok=True)
    model.save_pretrained("./results/baseline_model")
    tokenizer.save_pretrained("./results/baseline_model")

def main():
    print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    print("Loading dataset from local JSON files...")
    train_dataset = load_local_dataset("data/train.json")
    val_dataset = load_local_dataset("data/validation.json")

    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    tokenized_val = val_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    configs = [
        {"id": 1, "lr": 2e-5, "batch_size": 4, "epochs": 3},
        {"id": 2, "lr": 3e-5, "batch_size": 8, "epochs": 4},
        {"id": 3, "lr": 1e-4, "batch_size": 8, "epochs": 2}
    ]

    for cfg in configs:
        train_model(tokenized_train, tokenized_val, tokenizer,
                    cfg["id"], cfg["lr"], cfg["batch_size"], cfg["epochs"])

    save_baseline_model()

if __name__ == "__main__":
    main()
