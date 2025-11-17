import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    DebertaV2TokenizerFast, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    # EarlyStoppingCallback, # You can add this back if needed, but it's not strictly necessary with load_best_model_at_end=True
    set_seed # Important for reproducibility
)
from peft import LoraConfig, get_peft_model
import numpy as np
import evaluate
import os

# --- 1. CONFIGURATION ---
MODEL_ID = "microsoft/deberta-v3-base" 
TRAIN_DATA_FILE = "train.csv"
VAL_DATA_FILE = "validation.csv"
OUTPUT_DIR = "./deberta_mitre_adapter" 

# Set seed for reproducibility
set_seed(42)

# --- Define Labels for Classification ---
# Ensure ALL unique labels from your CSV files are listed here!
labels = ["Benign", "T1003.002", "T1087.001", "T1049"] 
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}
DELIMITER = " ### MITRE_ID: "

# --- 2. LOAD DATASET ---
print(f"Loading datasets from {TRAIN_DATA_FILE} and {VAL_DATA_FILE}...")
raw_datasets = DatasetDict({
    "train": load_dataset('csv', data_files=TRAIN_DATA_FILE, split="train"),
    "validation": load_dataset('csv', data_files=VAL_DATA_FILE, split="train"),
})

# --- 3. TOKENIZER & PREPROCESSING ---
# NOTE: DebertaV2TokenizerFast is the correct tokenizer for DeBERTa-v3
tokenizer = DebertaV2TokenizerFast.from_pretrained(MODEL_ID)

def preprocess_function(examples):
    # Tokenize the text column
    inputs = tokenizer(examples["text"], max_length=128, padding="max_length", truncation=True)
    
    # The 'label' column is used for training
    inputs["labels"] = examples["label"]
    
    return inputs

print("Preprocessing datasets...")
processed_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names)


# --- 4. MODEL LOADING (Sequence Classification) ---
print(f"Loading model for Sequence Classification: {MODEL_ID}...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Gradient checkpointing is enabled BEFORE LoRA is applied
model.gradient_checkpointing_enable() 
model.enable_input_require_grads() 
# NOTE: Manual freezing loop removed, as PEFT handles this.

# --- 5. LoRA CONFIG FOR SEQ_CLS (Updated Hyperparameters) ---
peft_config = LoraConfig(
    r=32,          # Adjusted: Lowered R from 64 to 32
    lora_alpha=64, # Adjusted: Set alpha = 2r for better scaling
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
    target_modules=["query_proj", "key_proj", "value_proj"]
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# --- 6. METRICS CALCULATION ---
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Use 'weighted' average for imbalanced multiclass classification
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels, average="weighted")
    recall = recall_metric.compute(predictions=predictions, references=labels, average="weighted")
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    
    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"],
    }

# --- 7. TRAINING ARGUMENTS (Updated Hyperparameters) ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5, # Adjusted: Increased from 3 to 5
    per_device_train_batch_size=32, 
    # CRITICAL CHANGE: Increased LR for LoRA training
    learning_rate=1e-4, 
    weight_decay=0.0, # Adjusted: Set to 0.0 to focus on adapter training
    bf16=True,
    
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy", # Keeping accuracy as the main metric for simplicity
    
    logging_steps=10,
    report_to="none",
    
    max_grad_norm=1.0, 
)

# --- 8. INITIALIZE TRAINER ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_datasets["train"],
    eval_dataset=processed_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# --- 9. TRAIN ---
print("Starting training for Sequence Classification...")
trainer.train()

# --- 10. SAVE ---
trainer.save_model(OUTPUT_DIR)
print(f"Training complete. Final model saved to {OUTPUT_DIR}")