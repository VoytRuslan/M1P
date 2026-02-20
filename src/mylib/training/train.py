import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from dataset.dataset import OCRDataset
from models.fusion import CrossModalFusion
from .losses import CTCLossWrapper, prepare_targets, get_input_lengths
from .metrics import cer, wer, greedy_decode
from .losses import TextEncoder

from torch_geometric.data import Data, Batch

from models.model import OCRModel, OCRModelCNNOnly
from torchvision import transforms
from tqdm import tqdm

def collate_fn(batch):
    images = []
    graphs = []
    texts = []

    for item in batch:
        images.append(item['image'])
        texts.append(item['text'])

        g = item['graph']
        data = Data(
            x=g['x'],
            edge_index=g['edge_index']
        )
        graphs.append(data)

    images = torch.stack(images)  # (B, 1, H, W)
    graphs = Batch.from_data_list(graphs)

    return {
        'images': images,
        'graphs': graphs,
        'texts': texts
    }
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = OCRDataset(
    txd_dir="../new_data",
    image_dir="../new_data",
    labels_file="../new_data/labels",
    transform=transforms.Compose([
        transforms.Resize((256, 2048)),
        transforms.ToTensor(),
    ]),
)

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn,
)
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?-()| '
encoder = TextEncoder(alphabet=alphabet)
model = OCRModel(
    num_classes=len(encoder.char2idx)
).to(device)

criterion = CTCLossWrapper(blank=0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def test(model, dataloader, encoder):
    model.eval()
    total_cer = 0
    total_wer = 0
    num_samples = 0
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            graphs = batch['graphs'].to(device)
            texts = batch['texts']
            targets, target_lengths = prepare_targets(texts, encoder)
            logits = model(images, graphs)
            input_lengths = get_input_lengths(logits).to(device)
            predictions = greedy_decode(logits, encoder)

            loss = criterion(logits, targets.to(device), input_lengths, target_lengths.to(device))
            total_loss += loss.item()

            for pred, target in zip(predictions, texts):
                print(pred, target)
                total_cer += cer(pred, target)
                total_wer += wer(pred, target)
                num_samples += 1

    avg_cer = total_cer / num_samples
    avg_wer = total_wer / num_samples
    avg_loss = total_loss / len(dataloader)
    print(f"Test CER: {avg_cer:.4f}, WER: {avg_wer:.4f}, Loss: {avg_loss:.4f}")

def train_epoch(model, dataloader, criterion, optimizer, encoder):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        images = batch['images'].to(device)
        graphs = batch['graphs'].to(device)
        texts = batch['texts']

        targets, target_lengths = prepare_targets(texts, encoder)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad()

        logits = model(images, graphs)

        input_lengths = get_input_lengths(logits).to(device)

        loss = criterion(logits, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Train Loss: {avg_loss:.4f}")
    test(model, test_loader, encoder)

for epoch in range(100):
    print(f"Epoch {epoch+1}/10")
    train_epoch(model, train_loader, criterion, optimizer, encoder)'''