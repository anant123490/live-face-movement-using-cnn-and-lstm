import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from models.caption_model import CaptionNet
from utils.text import Vocabulary


class CaptionDataset(Dataset):
    def __init__(self, dataframe, vocab, max_len=25):
        self.df = dataframe.reset_index(drop=True)
        self.vocab = vocab
        self.max_len = max_len
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        caption_ids = torch.tensor(
            self.vocab.encode(row["caption"], max_len=self.max_len), dtype=torch.long
        )
        return image, caption_ids


def train(args):
    os.makedirs(args.artifacts, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.csv)
    if not {"image_path", "caption"}.issubset(df.columns):
        raise ValueError("CSV must include 'image_path' and 'caption' columns")

    vocab = Vocabulary.build(df["caption"].tolist(), min_freq=args.min_freq)
    vocab_path = os.path.join(args.artifacts, "vocab.json")
    vocab.save(vocab_path)

    ds = CaptionDataset(df, vocab=vocab, max_len=args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = CaptionNet(
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        vocab_size=len(vocab),
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])
    params = list(model.decoder.parameters()) + list(model.encoder.fc.parameters()) + list(
        model.encoder.bn.parameters()
    )
    optimizer = torch.optim.Adam(params, lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for images, captions in tqdm(dl, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            images, captions = images.to(device), captions.to(device)
            logits = model(images, captions)
            targets = captions
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(dl))
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}")

    model_path = os.path.join(args.artifacts, "caption_model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embed_size": args.embed_size,
            "hidden_size": args.hidden_size,
            "vocab_size": len(vocab),
            "max_len": args.max_len,
        },
        model_path,
    )
    print(f"Saved model: {model_path}")
    print(f"Saved vocab: {vocab_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to captions CSV")
    parser.add_argument("--artifacts", default="artifacts", help="Output artifact directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--max-len", type=int, default=25)
    parser.add_argument("--min-freq", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
