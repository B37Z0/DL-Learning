import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

# ref: https://medium.com/@deepeshdeepakdd2/lenet-5-implementation-on-mnist-in-pytorch-c6f2ee306e37

### Preprocessing ###
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True), 
    v2.Pad(2),  # pad 28x28 -> 32x32
    v2.Normalize((0.1307,), (0.3081,))
])
training_data = datasets.MNIST(
    root="data", 
    train=True, 
    download=False, 
    transform=transform
)
test_data = datasets.MNIST(
    root="data", 
    train=False, 
    download=False, 
    transform=transform
)

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Intentionally inefficient syntax to work through each layer
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # (C=1, C'=6, K=5)
        self.conv2 = nn.Conv2d(6, 16, 5) # (C=6, C'=16, K=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # (C*H*W, 120 neurons)
        self.fc2 = nn.Linear(120, 84) # 84 neurons
        self.fc3 = nn.Linear(84, 10) # 10 classes

	# (N, 1, 32, 32) tensor input
    def forward(self, input):
	    # conv1:
	    # - (N, 6, 28, 28) tensor output 
	    # - ReLU 
        c1 = F.relu(self.conv1(input))
        # Max Pooling 1: 
        # - (N, 6, 14, 14) Tensor output
        p1 = F.max_pool2d(c1, (2, 2))
        # conv2:
        # - (N, 16, 10, 10) tensor output
        # - ReLU
        c2 = F.relu(self.conv2(p1))
        # Max Pooling 2:
        # - (N, 16, 5, 5) Tensor output
        p2 = F.max_pool2d(c2, 2)
        # flatten:
        # - (N, 16*5*5 = 400) Tensor output
        v0 = torch.flatten(p2, 1)
        # fc1: 
        # - (N, 120) Tensor output
        # - ReLU
        f1 = F.relu(self.fc1(v0))
        # fc2:
        # - (N, 84) Tensor output
        # - ReLU
        f2 = F.relu(self.fc2(f1))
        # fc3:
        # - (N, 10) Tensor OUTPUT
        output = self.fc3(f2)
        return output

model = LeNet5()
print(model)

### Optimizing ###
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.001)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):

        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # Update
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Training complete")

### Save/Load ###
PATH = "./LeNet5.pth"
torch.save(model.state_dict(), PATH)
print(f"Saved PyTorch Model State to {PATH}")

model = LeNet5()
model.load_state_dict(torch.load(PATH, weights_only=True))

### Evaluation ###
classes = ["0","1","2","3","4","5","6","7","8","9"]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():   
    pred = model(x.unsqueeze(0))  # add batch dim: (1, 32, 32) -> (1, 1, 32, 32)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
