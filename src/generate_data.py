import csv
import random
import itertools

# --- CONFIGURATION ---
TRAIN_FILE = "train.csv"
VALIDATION_FILE = "validation.csv"
TEST_FILE = "test.csv"
HELD_OUT_FILE = "held_out.csv"
LABEL = "T1003.002"

# Ratios: 80% Train, 10% Validation, 10% Test
TOTAL_EXAMPLES = 20000
TRAIN_COUNT = int(TOTAL_EXAMPLES * 0.8)
VALIDATION_COUNT = int(TOTAL_EXAMPLES * 0.1)
TEST_COUNT = int(TOTAL_EXAMPLES * 0.1)
HELD_OUT_COUNT = 2000

# --- DATA POOLS (Reduced for less diversity) ---

MALICIOUS_COMPONENTS = {
    "executables": ["reg.exe"],
    "actions": ["save", "export"],
    "hives": ["hklm\\sam", "hklm\\system", "hklm\\security"],
    "paths": [
        "C:\\sam.save",
        "C:\\Temp\\dump.save",
        "%TEMP%\\registry_backup.save",
    ],
    "wrappers": ["", "cmd.exe /c {}", "powershell -c {}"]
}

BENIGN_COMMANDS = [
    "reg query HKLM\\Software\\Microsoft", 
    "reg query HKCU\\Control Panel\\Desktop",
    "reg add HKCU\\Software\\MyApp /v Setting /d Value", 
    "reg delete HKCU\\Software\\MyApp /f",
    "reg compare HKLM\\Software\\MyCompany HKLM\\Software\\YourCompany",
    "reg copy HKCU\\Software\\MyApp HKCU\\Software\\MyApp_backup",
    "reg restore HKCU\\Software\\MyApp_backup",
    "reg unload HKLM\\TempHive",
    "reg query HKLM\\System\\CurrentControlSet\\Services",
    "reg query HKCU\\Environment",
    "reg add HKCU\\Environment /v MY_VAR /d C:\\MyPath",
    "powershell -c reg query HKCU\\Volatile Environment"
]

# --- DATA GENERATION FUNCTIONS ---

def get_random_malicious_command():
    """Generates a single, random malicious command and its label."""
    comp = MALICIOUS_COMPONENTS
    command_str = f"{random.choice(comp['executables'])} {random.choice(comp['actions'])} {random.choice(comp['hives'])} {random.choice(comp['paths'])}"
    
    wrapper = random.choice(comp['wrappers'])
    if wrapper:
        command_str = wrapper.format(command_str)
        
    return command_str, 1 # Return command and label 1

def generate_benign_command():
    """Generates a single random benign command and its label."""
    command = random.choice(BENIGN_COMMANDS)
    return command, 0 # Return command and label 0

def create_dataset(file_path, num_malicious, num_benign):
    """Creates a balanced and shuffled dataset and writes it to a CSV file."""
    print(f"Generating {file_path} with {num_malicious} malicious and {num_benign} benign examples...")
    data = []
    for _ in range(num_malicious):
        data.append(get_random_malicious_command())
    for _ in range(num_benign):
        data.append(generate_benign_command())
        
    random.shuffle(data)
    
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])  # Write header
        writer.writerows(data)  # Write data rows
    print(f"Successfully created {file_path}.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("--- Starting Final Dataset Generation ---")
    
    # Calculate counts for a 50/50 split within each dataset
    train_mal_count = train_ben_count = TRAIN_COUNT // 2
    val_mal_count = val_ben_count = VALIDATION_COUNT // 2
    test_mal_count = test_ben_count = TEST_COUNT // 2
    held_out_mal_count = held_out_ben_count = HELD_OUT_COUNT // 2
    
    # Generate each dataset directly
    create_dataset(TRAIN_FILE, train_mal_count, train_ben_count)
    create_dataset(VALIDATION_FILE, val_mal_count, val_ben_count)
    create_dataset(TEST_FILE, test_mal_count, test_ben_count)
    create_dataset(HELD_OUT_FILE, held_out_mal_count, held_out_ben_count)
    
    print("\n--- Dataset Generation Complete. All files are correctly formatted and balanced. ---")
