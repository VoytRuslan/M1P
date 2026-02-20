import torch
import torch.nn as nn
import math


class CrossModalFusion(nn.Module):
    def __init__(self, d_v, d_h, d_k):
        super().__init__()

        self.W_Q = nn.Linear(d_v, d_k)
        self.W_K = nn.Linear(d_h, d_k)
        self.W_V = nn.Linear(d_h, d_v)

        self.W_O = nn.Linear(2 * d_v, d_v)

    def forward(self, V, H, batch_graph=None):
        if batch_graph is not None:
            batch_idx = batch_graph.batch
            H_split = []
            for i in range(V.size(0)):
                H_split.append(H[batch_idx == i])
        else:
            H_split = [H] * V.size(0)

        outputs = []

        for i in range(V.size(0)):
            v = V[i]
            h = H_split[i]

            if h.size(0) == 0:
                outputs.append(v)
                continue

            Q = self.W_Q(v)
            K = self.W_K(h)
            V_val = self.W_V(h)

            attn = torch.softmax(
                Q @ K.T / math.sqrt(K.size(-1)),
                dim=-1
            )

            H_tilde = attn @ V_val
            U = torch.cat([v, H_tilde], dim=-1)
            U_proj = self.W_O(U)
            outputs.append(U_proj)

        return torch.stack(outputs)
