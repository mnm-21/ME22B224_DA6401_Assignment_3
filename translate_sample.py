import torch
from model import Transformer

def translate():
    # CPU for a quick single inference
    device = torch.device("cpu")
    
    print("Initializing model and loading checkpoint...")
    # The Transformer class automatically loads best_checkpoint.pt and vocab files on init
    # If it fails to find them, it downloads from gdown as per model.py logic
    model = Transformer().to(device)
    model.eval()
    
    test_sentences = [
        "Ein Mann sitzt auf einer Bank im Park und liest ein Buch.",
        "Ein kleiner Junge spielt im Sand.",
        "Zwei Hunde laufen über eine grüne Wiese.",
        "Eine Frau in einem roten Kleid geht die Straße entlang.",
        "Ein Kind springt in einen Pool."
    ]
    
    print("\n" + "="*50)
    for de_sentence in test_sentences:
        with torch.no_grad():
            en_translation = model.infer(de_sentence)
            
        print(f"DE: {de_sentence}")
        print(f"EN: {en_translation}")
        print("-" * 50)

if __name__ == "__main__":
    translate()
