import torch
import torch.nn as nn


class CTCLossWrapper(nn.Module):
    def __init__(self, blank=0):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, zero_infinity=True)

    def forward(self, logits, targets, input_lengths, target_lengths):
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = self.ctc(
            log_probs,
            targets,
            input_lengths,
            target_lengths
        )

        return loss

class TextEncoder:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.char2idx = {c: i + 1 for i, c in enumerate(alphabet)}
        self.char2idx["<blank>"] = 0

        self.idx2char = {i: c for c, i in self.char2idx.items()}

    def encode(self, text):
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def decode(self, indices):
        chars = []
        prev = None

        for i in indices:
            if i != 0 and i != prev:
                chars.append(self.idx2char[i])
            prev = i

        return "".join(chars)

def prepare_targets(texts, encoder):
    targets = []
    target_lengths = []

    for t in texts:
        encoded = encoder.encode(t)
        targets.extend(encoded)
        target_lengths.append(len(encoded))

    targets = torch.tensor(targets, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return targets, target_lengths

def get_input_lengths(logits):
    T, B, _ = logits.shape
    return torch.full(size=(B,), fill_value=T, dtype=torch.long)

