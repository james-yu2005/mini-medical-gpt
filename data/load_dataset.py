from pathlib import Path

from datasets import load_dataset

CORPUS_PATH = Path(__file__).resolve().parent / "input.txt"


def main() -> None:
    print("Downloading medical dataset...")
    dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
    text = "\n".join(dataset["output"][:50000])

    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_PATH.write_text(text, encoding="utf-8")
    print(f"Saved {len(text):,} characters to {CORPUS_PATH}")


if __name__ == "__main__":
    main()
