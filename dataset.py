import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import spacy

PAD_IDX = 1
SPECIAL_TOKENS = {"<unk>": 0, "<pad>": 1, "<sos>": 2, "<eos>": 3}


class Multi30kDataset(Dataset):
    """Loads and tokenizes the Multi30k German-English dataset."""

    def __init__(self, split="train", src_vocab=None, tgt_vocab=None):
        self.split = split
        ds = load_dataset("bentrevett/multi30k")
        self.data = ds[split]

        self.src_tokenizer = spacy.load("de_core_news_sm")
        self.tgt_tokenizer = spacy.load("en_core_web_sm")

        if src_vocab is None or tgt_vocab is None:
            self.src_vocab, self.tgt_vocab = self.build_vocab()
        else:
            self.src_vocab, self.tgt_vocab = src_vocab, tgt_vocab

        self.processed = self.process_data()

    def tokenize(self, text, tokenizer):
        return [tok.text.lower() for tok in tokenizer(text)]

    def build_vocab(self):
        """Creates token-to-index mappings for both languages."""
        src_vocab, tgt_vocab = dict(SPECIAL_TOKENS), dict(SPECIAL_TOKENS)
        for example in self.data:
            for tok in self.tokenize(example["de"], self.src_tokenizer):
                if tok not in src_vocab:
                    src_vocab[tok] = len(src_vocab)
            for tok in self.tokenize(example["en"], self.tgt_tokenizer):
                if tok not in tgt_vocab:
                    tgt_vocab[tok] = len(tgt_vocab)
        return src_vocab, tgt_vocab

    def process_data(self):
        """Converts raw text sentences into integer tensors."""
        processed = []
        sos, eos, unk = (
            SPECIAL_TOKENS["<sos>"],
            SPECIAL_TOKENS["<eos>"],
            SPECIAL_TOKENS["<unk>"],
        )
        for example in self.data:
            src_ids = (
                [sos]
                + [
                    self.src_vocab.get(t, unk)
                    for t in self.tokenize(example["de"], self.src_tokenizer)
                ]
                + [eos]
            )
            tgt_ids = (
                [sos]
                + [
                    self.tgt_vocab.get(t, unk)
                    for t in self.tokenize(example["en"], self.tgt_tokenizer)
                ]
                + [eos]
            )
            processed.append(
                (
                    torch.tensor(src_ids, dtype=torch.long),
                    torch.tensor(tgt_ids, dtype=torch.long),
                )
            )
        return processed

    def __len__(self):
        return len(self.processed)

    def __getitem__(self, idx):
        return self.processed[idx]


def collate_fn(batch):
    """Pads sentences to the same length for batch processing."""
    src_batch, tgt_batch = zip(*batch)
    return pad_sequence(
        src_batch, batch_first=True, padding_value=PAD_IDX
    ), pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
