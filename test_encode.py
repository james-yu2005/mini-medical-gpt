"""Interactive BPE encode/decode tester. Run: python test_encode.py"""

from tokenizer.corpus import load_tokenizer

tok = load_tokenizer()

while True:
    text = input("Enter text (or blank to quit): ").strip()
    if not text:
        break
    ids = tok.encode(text)
    print("IDs:   ", ids)
    print("Tokens:", [tok.token_to_str(i) for i in ids])
    print("Back:  ", tok.decode(ids))
    print()
