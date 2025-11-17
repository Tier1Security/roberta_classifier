import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer
)
from evaluate import load
from typing import Dict, Any, List

# --- CONFIGURATION ---
# Assuming 'final_deberta_model' folder and 'test.csv' are in the 
# same directory as this script.
FINAL_MODEL_DIR = "./final_deberta_model" 
TEST_DATASET_FILE = "./test.csv" 
MAX_SEQ_LENGTH = 128 
LABEL_COLUMN_NAME = "label" 

# --- Path Resolution ---
# We use abspath to ensure absolute paths are passed to Hugging Face 
# which avoids the 'Repo id' validation error we saw previously.
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    FINAL_MODEL_PATH = os.path.abspath(os.path.join(script_dir, FINAL_MODEL_DIR))
    TEST_DATASET_PATH = os.path.abspath(os.path.join(script_dir, TEST_DATASET_FILE))
except NameError:
    # Fallback if __file__ is not defined
    FINAL_MODEL_PATH = FINAL_MODEL_DIR
    TEST_DATASET_PATH = TEST_DATASET_FILE

print(f"Loading merged classification model from resolved path: {FINAL_MODEL_PATH}...")

# --- 1. Load Model, Tokenizer, and Data ---
# Using the resolved absolute path and ensuring we only look locally.
tokenizer = AutoTokenizer.from_pretrained(
    FINAL_MODEL_PATH,
    local_files_only=True
)

# Load model onto GPU with BF16
model = AutoModelForSequenceClassification.from_pretrained(
    FINAL_MODEL_PATH,
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    local_files_only=True
)
model.eval()

# Load test dataset
print(f"Loading test data from resolved path: {TEST_DATASET_PATH}...")
# Assuming your CSV has 'text' (input) and 'label' (integer class ID) columns
# Load dataset using the resolved absolute path
test_dataset = load_dataset('csv', data_files=TEST_DATASET_PATH, split="train")

# Check if the required label column exists
if LABEL_COLUMN_NAME not in test_dataset.column_names:
    raise ValueError(
        f"Test dataset must contain a column named '{LABEL_COLUMN_NAME}' for classification labels. "
        "Please check your 'test.csv' file."
    )

# --- 2. Tokenization and Preprocessing ---
def tokenize_function(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """Tokenizes text and prepares classification labels."""
    # Tokenize the input text
    tokenized = tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=MAX_SEQ_LENGTH,
    )
    
    # For Sequence Classification, the labels are the integer class IDs (0, 1, 2, 3...)
    tokenized["labels"] = examples[LABEL_COLUMN_NAME]
    return tokenized

print("Tokenizing and mapping test dataset...")
tokenized_test_dataset = test_dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=test_dataset.column_names, 
)

# --- 3. Define Metric Computation ---
# Load the accuracy metric
accuracy_metric = load("accuracy")

def compute_metrics(eval_pred: tuple) -> Dict[str, float]:
    """Computes sequence-level classification accuracy."""
    logits, labels = eval_pred
    
    # For classification, we find the index of the highest logit for each sample in the batch.
    predictions = np.argmax(logits, axis=1)
    
    # Labels are already the integer class IDs.
    
    return accuracy_metric.compute(
        predictions=predictions, 
        references=labels,
    )

# --- 4. Setup Trainer and Run Evaluation ---
training_args = TrainingArguments(
    output_dir="./evaluation_results",
    per_device_eval_batch_size=16, 
    dataloader_drop_last=False,
    bf16=True, 
    report_to="none", 
)

# Use the tokenized test set for final evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("\n--- Running Evaluation on Test Data ---")
evaluation_metrics = trainer.evaluate()

# --- 5. Print Results ---
print("\n--- ✅ Final Testing Metrics ---")
for key, value in evaluation_metrics.items():
    print(f"  {key}: {value:.4f}")
print("---------------------------------")