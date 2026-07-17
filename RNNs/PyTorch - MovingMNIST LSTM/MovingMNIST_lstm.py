import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision as tv
from torchvision.datasets import MovingMNIST
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
MovingMNIST Next-Frame Prediction with ResNet-18 Encoder + LSTM + ConvTranspose Decoder
---------------------------------------------------------------------------------------
Train a video prediction model on the MovingMNIST dataset - 20-frame sequences of two 
digits bouncing around a 64x64 canvas. The LSTM model observes the first 10 frames for
context and autoregressively predicts the next 10.

Architecture
---------------------------------------------------------------------------------------
1. Encoder (frozen): Pretrained ResNet-18 (minus classification head) to encode
   grayscale framed (repeated to 3 channels) into a 512-D feature vector.

2. LSTM: Single-layer LSTM builds hidden state from 10 context embeddings, then runs 
   autoregressively for 10 steps to produce 10 future embeddings.

3. Decoder: Stack of ConvTranspose2d layers to upsample 512x1x1 embeddings back to a 
   1x64x64 image (Sigmoid-activated to match [0,1] pixel range).

Results (see Figure_1.png)
---------------------------------------------------------------------------------------
- The model captures the general position and motion trajectory of the digits; the 
  bright blobs roughly appear in the correct locations and drift in the right direction
  across the 10 predicted frames.
- Predictions are blurry and lack fine details; overlapping digits merge into a single
  blob. This is expected for pixel-level MSE loss and a limitation of the design.
- Validation loss plateaus around ~0.031 after ~8 epochs, suggesting the model has 
  learned what it can within its capacity.

Future Improvements
---------------------------------------------------------------------------------------
- Replace MSE with something like perceptual loss
- Use a ConvLSTM that operates in a spatial feature-map space (convention)
- Use skip connections and/or a distinct LSTM encoder and decoder
"""

## Neural Network ##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Backbone
resnet_net = tv.models.resnet18(weights="DEFAULT")
backbone = nn.Sequential(*list(resnet_net.children())[:-1]).to(device)
backbone.eval()
for p in backbone.parameters():
    p.requires_grad = False

# LSTM Model
lstm = nn.LSTM(input_size=512, hidden_size=512).to(device)

# Decoder 
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 1, 0), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)
decoder = Decoder().to(device)


## Optimization ##
# Data is split by frame windows, not sequences - sample a validation set and slice context/target manually
dataset = MovingMNIST(root=".", split=None, download=True)
val_size = int(0.1 * len(dataset))
train_subset, val_subset = random_split(dataset, [len(dataset) - val_size, val_size])

train_dataloader = DataLoader(train_subset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_subset, batch_size=8, shuffle=False)

optimizer = torch.optim.Adam(list(lstm.parameters()) + list(decoder.parameters()), lr=1e-3)
criterion = nn.MSELoss()


def forward_pass(batch, backbone, lstm, decoder, device, pred_len=10):
    # Normalize pixels to (0, 1] to match Sigmoid output
    batch = batch.float().to(device) / 255.0
    context = batch[:, :10]
    target = batch[:, 10:]

    context = torch.repeat_interleave(context, repeats=3, dim=2).permute(1,0,2,3,4)
    target = target.permute(1,0,2,3,4)

    with torch.no_grad():
        L, N, C, H, W = context.shape
        context_embedding = backbone(context.reshape(-1, C, H, W)).reshape(L, N, -1)

    # Predictions with context frames
    _, (h, c) = lstm(context_embedding)

    # Autoregressive predictions w/ preserved context state
    ar_input = context_embedding[-1:]
    ar_outputs = []
    for _ in range(pred_len):
        ar_output, (h, c) = lstm(ar_input, (h, c))
        ar_outputs.append(ar_output)
        ar_input = ar_output

    prediction_embedding = torch.cat(ar_outputs, dim=0)
    decoder_input = prediction_embedding.reshape(-1, 512, 1, 1)
    decoded_image = decoder(decoder_input).reshape(pred_len, N, 1, 64, 64)
    return decoded_image, target

@torch.no_grad()
def validate(lstm, decoder, backbone, dataloader, criterion, device):
    lstm.eval()
    decoder.eval()
    epoch_loss = 0.0

    for batch in dataloader:
        decoded_image, target = forward_pass(batch, backbone, lstm, decoder, device)
        epoch_loss += criterion(decoded_image, target).item()

    lstm.train()
    decoder.train()
    return epoch_loss / len(dataloader)


def train_model(lstm, decoder, backbone, dataloader, optimizer, criterion, device, num_epochs=10, val_loader=None):
    for epoch in range(num_epochs):
        lstm.train()
        decoder.train()
        epoch_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Forward pass
            decoded_image, target = forward_pass(batch, backbone, lstm, decoder, device)
            loss = criterion(decoded_image, target)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Update
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        log = f"Epoch {epoch+1}: train {avg_loss:.4f}"

        if val_loader is not None:
            val_loss = validate(lstm, decoder, backbone, val_loader, criterion, device)
            log += f" | val {val_loss:.4f}"

        print(log)

        torch.save({
            "epoch": epoch,
            "lstm_state": lstm.state_dict(),
            "decoder_state": decoder.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": avg_loss,
        }, f"checkpoint_epoch{epoch+1}.pt")


@torch.no_grad()
def visualize(lstm, decoder, backbone, dataloader, device, num_frames=10):
    lstm.eval()
    decoder.eval()
    batch = next(iter(dataloader))
    decoded_image, target = forward_pass(batch, backbone, lstm, decoder, device)

    # First sequence in batch -> (L, H, W)
    pred = decoded_image[:, 0, 0].cpu()
    gt = target[:, 0, 0].cpu()

    fig, axes = plt.subplots(2, num_frames, figsize=(2*num_frames, 4))
    for t in range(num_frames):
        axes[0, t].imshow(gt[t], cmap='gray'); axes[0, t].axis('off')
        axes[0, t].set_title(f'Target Img {t+1}')

        axes[1, t].imshow(pred[t], cmap='gray'); axes[1, t].axis('off')
        axes[1, t].set_title(f'Decoded Img {t+1}')
    plt.tight_layout()
    plt.show()

    lstm.train()
    decoder.train()


train_model(lstm, decoder, backbone, train_dataloader, optimizer, criterion, device,
            num_epochs=10, val_loader=val_dataloader)

visualize(lstm, decoder, backbone, val_dataloader, device)

'''
Epoch 1: train 0.0420 | val 0.0349
Epoch 2: train 0.0344 | val 0.0341
Epoch 3: train 0.0337 | val 0.0334
Epoch 4: train 0.0330 | val 0.0330
Epoch 5: train 0.0325 | val 0.0325
Epoch 6: train 0.0318 | val 0.0323
Epoch 7: train 0.0314 | val 0.0321
Epoch 8: train 0.0309 | val 0.0318
Epoch 9: train 0.0304 | val 0.0315
Epoch 10: train 0.0301 | val 0.0316
'''