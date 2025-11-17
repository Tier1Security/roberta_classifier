import torch
from transformers import AutoModelForSequenceClassification, DebertaV2TokenizerFast
from peft import PeftModel
import os

# --- CONFIGURATION ---
BASE_MODEL_ID = "microsoft/deberta-v3-base"
ADAPTER_PATH = "models/final_deberta_model"
MERGED_MODEL_OUTPUT_DIR = "models/merged_deberta_model"

# Define Labels for Classification (Must match training)
labels = ["Benign", "T1003.002", "T1087.001", "T1049"]
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}

# --- 1. LOAD BASE MODEL ---
print(f"Loading base model: {BASE_MODEL_ID}...")
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_ID,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# --- 2. LOAD TOKENIZER ---
print(f"Loading tokenizer for {BASE_MODEL_ID}...")
tokenizer = DebertaV2TokenizerFast.from_pretrained(BASE_MODEL_ID)


# --- 3. LOAD AND MERGE ADAPTER ---
print(f"Loading LoRA adapter from: {ADAPTER_PATH}...")
# Attach the saved LoRA weights to the base model
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("Merging adapter weights into the base model...")
# Merge the adapter weights into the base model weights
model = model.merge_and_unload()
print("Merge complete.")


# --- 4. SAVE MERGED MODEL AND TOKENIZER ---
print(f"Saving merged model to: {MERGED_MODEL_OUTPUT_DIR}...")
os.makedirs(MERGED_MODEL_OUTPUT_DIR, exist_ok=True)
model.save_pretrained(MERGED_MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MERGED_MODEL_OUTPUT_DIR)

print(f"\n--- ✅ Merged model and tokenizer saved successfully to {MERGED_MODEL_OUTPUT_DIR} ---")
