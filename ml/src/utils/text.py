import json
import re
from collections import Counter


SPECIAL_TOKENS = ["<pad>", "<start>", "<end>", "<unk>"]


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class Vocabulary:
    def __init__(self, stoi=None, itos=None):
        self.stoi = stoi or {}
        self.itos = itos or {}

    @classmethod
    def build(cls, captions, min_freq=2):
        counter = Counter()
        for cap in captions:
            counter.update(clean_text(cap).split())

        words = [w for w, c in counter.items() if c >= min_freq]
        idx_to_token = SPECIAL_TOKENS + sorted(words)
        stoi = {tok: idx for idx, tok in enumerate(idx_to_token)}
        itos = {idx: tok for tok, idx in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    def encode(self, text: str, max_len: int):
        tokens = clean_text(text).split()
        ids = [self.stoi["<start>"]]
        ids.extend(self.stoi.get(tok, self.stoi["<unk>"]) for tok in tokens)
        ids.append(self.stoi["<end>"])
        ids = ids[:max_len]
        pad_id = self.stoi["<pad>"]
        ids += [pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids):
        words = []
        for idx in ids:
            token = self.itos.get(int(idx), "<unk>")
            if token in ("<pad>", "<start>"):
                continue
            if token == "<end>":
                break
            words.append(token)
        return " ".join(words).strip()

    def __len__(self):
        return len(self.stoi)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi, "itos": self.itos}, f, ensure_ascii=True, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # JSON keys for itos come back as strings
        itos = {int(k): v for k, v in data["itos"].items()}
        return cls(stoi=data["stoi"], itos=itos)
