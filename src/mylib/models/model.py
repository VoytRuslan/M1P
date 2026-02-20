import torch
import torch.nn as nn

from .cnn import CNNEncoder
from .gnn import GNNEncoder
from .fusion import CrossModalFusion


class OCRModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = CNNEncoder(output_dim=512)
        self.gnn = GNNEncoder(input_dim=11, hidden_dim=256)
        self.fusion = CrossModalFusion(
            d_v=512,
            d_h=256,
            d_k=128
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, image, graph):
        V = self.cnn(image)
        H = self.gnn(graph)
        U = self.fusion(V, H, graph)
        logits = self.classifier(U)
        return logits.permute(1, 0, 2)

class OCRModelCNNOnly(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = CNNEncoder(output_dim=512)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, image, graph=None):
        V = self.cnn(image)
        logits = self.classifier(V)
        return logits.permute(1, 0, 2)


class OCRModelGNNOnly(nn.Module):
    def __init__(self, num_classes, max_seq_len=128):
        super().__init__()
        self.gnn = GNNEncoder(input_dim=11, hidden_dim=256)
        self.max_seq_len = max_seq_len
        self.projection = nn.Linear(256, 512)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, image=None, graph=None):
        H = self.gnn(graph)
        batch_idx = graph.batch
        B = batch_idx.max().item() + 1
        outputs = []
        for i in range(B):
            h_i = H[batch_idx == i]
            if h_i.size(0) == 0:
                h_i = torch.zeros(1, 256, device=H.device)
            h_agg = h_i.mean(dim=0, keepdim=True)
            h_seq = h_agg.repeat(self.max_seq_len, 1)
            outputs.append(h_seq)
        U = torch.stack(outputs)
        U = self.projection(U)
        logits = self.classifier(U)
        return logits.permute(1, 0, 2)
