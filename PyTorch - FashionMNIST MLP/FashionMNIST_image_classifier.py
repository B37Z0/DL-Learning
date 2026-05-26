import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

# Complement to CIFAR10 Image Classifier.
# FasionMNIST has a more preferable implementation.
# Recommended to review workflow in the future. 

### Preprocessing ###
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True), 
    v2.Normalize((0.2860,), (0.3530,))
])
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform,
)

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


### Neural Network ###
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Flatten is stylistic
        # Can also call torch.flatten() in forward()
        self.flatten = nn.Flatten()
        # nn.Sequential is a container that automatically chains layers together
        # Don't have to manually pass the output of each layer into the next in forward()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), # 1st hidden layer
            nn.ReLU(),
            nn.Linear(512, 512), # 2nd hidden layer
            nn.ReLU(),
            nn.Linear(512, 10)  # output layer
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# # Commented out because I'm using cpu
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# print(f"Using {device} device")
model = NeuralNetwork() #.to(device)
print(model)


### Training/Testing ###
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # model.train() sets model to training mode - don't forget this!
    model.train()
    # enumerate() gives us the batch index & batch of data
    for batch, (X, y) in enumerate(dataloader):
        # X, y = X.to(device), y.to(device) # Commented out because I'm using cpu

        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # Update
        optimizer.step()

        # Print every 100 batches
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # model.eval() sets model to evaluation mode - don't forget this!
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            # X, y = X.to(device), y.to(device) # Commented out because I'm using cpu
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
PATH = "./fashionMNIST_net.pth"

torch.save(model.state_dict(), PATH)
print(f"Saved PyTorch Model State to {PATH}")

model = NeuralNetwork()
model.load_state_dict(torch.load(PATH, weights_only=True))


### Eval ###
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    # x = x.to(device) # Commented out because I'm using cpu
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')