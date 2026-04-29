import torch
import torch.nn as nn
import torchvision.models as models


class CNNEncoder(nn.Module):
    def __init__(self, embed_size=256):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modules = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.fc = nn.Linear(backbone.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.backbone(images).flatten(1)
        features = self.fc(features)
        return self.bn(features)


class LSTMDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        if num_layers <= 1:
            dropout = 0.0
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout
        )
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        features = features.unsqueeze(1)
        inputs = torch.cat((features, embeddings), dim=1)
        outputs, _ = self.lstm(inputs)
        return self.linear(outputs)

    def sample(self, features, max_len, start_id, end_id):
        states = None
        sampled_ids = [start_id]
        inputs = features.unsqueeze(1)
        for _ in range(max_len - 1):
            hiddens, states = self.lstm(inputs, states)
            logits = self.linear(hiddens.squeeze(1))
            predicted = torch.argmax(logits, dim=1)
            token_id = predicted.item()
            sampled_ids.append(token_id)
            if token_id == end_id:
                break
            inputs = self.embed(predicted).unsqueeze(1)
        return sampled_ids


class CaptionNet(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        self.encoder = CNNEncoder(embed_size=embed_size)
        self.decoder = LSTMDecoder(
            embed_size=embed_size, hidden_size=hidden_size, vocab_size=vocab_size
        )

    def forward(self, images, captions):
        features = self.encoder(images)
        return self.decoder(features, captions)
