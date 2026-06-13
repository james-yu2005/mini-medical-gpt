import urllib.request
from datasets import load_dataset

# Load a medical flashcard/wikidoc dataset and concatenate it into a single string
print("Downloading medical dataset...")
dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")

# Extract the 'output' column which contains dense medical explanations
text = "\n".join(dataset['output'][:50000]) # Take first 50k for a manageable size

with open("input.txt", "w", encoding="utf-8") as f:
    f.write(text)

print(f"Dataset saved. Length: {len(text)} characters.")