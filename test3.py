import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from model import EncoderCNN, DecoderRNN
from dataset import CocoDataset  # Zakładając, że plik dataset.py jest dostępny
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge


def get_loader(root_folder, annotation_file, transform, batch_size=32, num_workers=2, shuffle=True, pin_memory=True):
    dataset = CocoDataset(root_folder, annotation_file, transform=transform)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory
    )
    return loader, dataset


# NOWA FUNKCJA do wczytywania tylko wybranych obrazów
def get_specific_loader(root_folder, annotation_file, transform, image_ids, batch_size=32, num_workers=2, shuffle=False,
                        pin_memory=True):
    """
    Tworzy DataLoader dla określonych identyfikatorów obrazów.
    """
    dataset = CocoDataset(root_folder, annotation_file, transform=transform, image_ids=image_ids)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,  # Zazwyczaj False dla ewaluacji, aby zachować kolejność
        pin_memory=pin_memory
    )
    return loader, dataset


def evaluate(model, data_loader, device, vocab):
    model.eval()
    references_corpus = []
    candidates_corpus = []

    rouge = Rouge()
    rouge_scores = []
    meteor_scores = []

    with torch.no_grad():
        for idx, (imgs, captions) in tqdm(enumerate(data_loader), total=len(data_loader), desc="Evaluating"):
            imgs = imgs.to(device)

            # Generowanie podpisów
            features = model.encoder(imgs)
            predicted_ids = model.decoder.sample(features)

            # Konwersja ID na słowa
            predicted_captions = []
            for ids in predicted_ids:
                predicted_captions.append(
                    [vocab.itos[i] for i in ids.cpu().numpy() if
                     i not in [vocab.stoi["<PAD>"], vocab.stoi["<SOS>"], vocab.stoi["<EOS>"]]]
                )

            # Przygotowanie referencji (prawdziwych podpisów)
            # captions ma wymiar (batch_size, num_captions, seq_length)
            # Musimy dostosować go do formatu NLTK: lista list list słów
            for i in range(imgs.size(0)):
                img_captions = []
                for cap_idx in range(captions.shape[1]):
                    caption_tokens = []
                    for token_id in captions[i, cap_idx].cpu().numpy():
                        if token_id not in [vocab.stoi["<PAD>"], vocab.stoi["<SOS>"], vocab.stoi["<EOS>"]]:
                            caption_tokens.append(vocab.itos[token_id])
                    if caption_tokens:  # Dodaj tylko, jeśli nie jest puste
                        img_captions.append(caption_tokens)

                if img_captions:
                    references_corpus.append(img_captions)
                    candidate = predicted_captions[i]
                    candidates_corpus.append(candidate)

                    # Obliczanie ROUGE i METEOR dla każdej pary
                    try:
                        # Przygotowanie stringów dla ROUGE i METEOR
                        candidate_str = ' '.join(candidate)
                        references_str = [' '.join(ref) for ref in img_captions]

                        # ROUGE
                        scores = rouge.get_scores(candidate_str, references_str[0])  # Porównanie z pierwszą referencją
                        rouge_scores.append(scores[0])

                        # METEOR
                        meteor_scores.append(meteor_score(references_str, candidate_str))

                    except ValueError:
                        # Puste predykcje mogą powodować błąd
                        pass

    # Obliczanie BLEU scores
    bleu1 = corpus_bleu(references_corpus, candidates_corpus, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references_corpus, candidates_corpus, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references_corpus, candidates_corpus, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(references_corpus, candidates_corpus, weights=(0.25, 0.25, 0.25, 0.25))

    # Obliczanie średniego ROUGE
    avg_rouge_l_f = sum(s['rouge-l']['f'] for s in rouge_scores) / len(rouge_scores) if rouge_scores else 0

    # Obliczanie średniego METEOR
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0

    print(f"Corpus BLEU-1: {bleu1:.4f}")
    print(f"Corpus BLEU-2: {bleu2:.4f}")
    print(f"Corpus BLEU-3: {bleu3:.4f}")
    print(f"Corpus BLEU-4: {bleu4:.4f}")
    print(f"Average METEOR: {avg_meteor:.4f}")
    print(f"Average ROUGE-L (F1): {avg_rouge_l_f:.4f}")

    return bleu1, bleu2, bleu3, bleu4


# Musisz również zmodyfikować klasę CocoDataset, aby akceptowała listę `image_ids`
# Dodaję tę modyfikację tutaj dla kompletności. Zastosuj ją w pliku `dataset.py`
class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, json, transform=None, image_ids=None):
        self.root = root
        self.coco = COCO(json)

        if image_ids:
            # Filtruj self.ids, aby zawierały tylko te z podanej listy
            all_ids = self.coco.getImgIds()
            self.ids = [img_id for img_id in all_ids if img_id in image_ids]
        else:
            self.ids = list(sorted(self.coco.getImgIds()))

        self.transform = transform
        self.vocab = self.build_vocab()

    def __getitem__(self, index):
        # Implementacja __getitem__ ...
        # ...
        # Musisz dokończyć implementację bazując na swoim oryginalnym kodzie
        pass

    def __len__(self):
        return len(self.ids)

    def build_vocab(self):
        # Implementacja budowania słownika...
        # ...
        pass


def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # --- TUTAJ WPROWADŹ LISTĘ ID OBRAZÓW ---
    # Przykładowa lista ID z COCO val2014
    specific_image_ids = [397133, 397136, 397147, 397148, 397151]

    # Użyj nowej funkcji, aby załadować tylko wybrane obrazy
    data_loader, dataset = get_specific_loader(
        root_folder="coco/val2014",
        annotation_file="coco/annotations/captions_val2014.json",
        transform=transform,
        image_ids=specific_image_ids,
        num_workers=2,
        batch_size=10  # Można dostosować
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Załaduj słownik (zakładając, że jest zapisany)
    # vocab = dataset.vocab
    # vocab_size = len(vocab)
    # Zastąp poniższe prawdziwym ładowaniem słownika
    class MockVocab:
        def __init__(self):
            self.itos = {i: str(i) for i in range(10000)}
            self.stoi = {str(i): i for i in range(10000)}
            self.stoi.update({"<PAD>": 0, "<SOS>": 1, "<EOS>": 2})

    vocab = MockVocab()
    vocab_size = 10000

    # Inicjalizacja modelu
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)

    model = nn.ModuleDict({'encoder': encoder, 'decoder': decoder})

    # Załaduj wytrenowane wagi
    try:
        model.load_state_dict(torch.load("image_captioning_model.pth", map_location=device))
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print("Error: Model weights not found. Make sure 'image_captioning_model.pth' is in the correct directory.")
        return

    # Ewaluacja modelu
    evaluate(model, data_loader, device, vocab)


if __name__ == "__main__":
    main()
