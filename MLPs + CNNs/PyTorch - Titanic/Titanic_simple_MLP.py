import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


### Preprocessing ###
path = './data/'
df = pd.read_csv(path + 'train.csv')
tst_df = pd.read_csv(path + "test.csv")

# Save PassengerIds now — they'll be needed for the Kaggle submission CSV
# but aren't used as features, so we don't want them in tst_df during preprocessing.
passenger_ids = tst_df["PassengerId"].copy()

# Feature engineering ref:
# https://www.kaggle.com/code/jhoward/why-you-should-use-a-framework/notebook
def add_features(df):
    df['LogFare'] = np.log1p(df['Fare'])
    df['Deck'] = df.Cabin.str[0].map(dict(A="ABC", B="ABC", C="ABC", D="DE", E="DE", F="FG", G="FG"))
    df['Family'] = df.SibSp + df.Parch
    df['Alone'] = df.Family == 0
    df['TicketFreq'] = df.groupby('Ticket')['Ticket'].transform('count')
    df['Title'] = df.Name.str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
    df['Title'] = df.Title.map(dict(Mr="Mr", Miss="Miss", Mrs="Mrs", Master="Master"))

add_features(df)
add_features(tst_df)

cat_cols = ["Sex", "Pclass", "Embarked", "Deck", "Title"]
cont_cols = ["Age", "SibSp", "Parch", "LogFare", "Alone", "TicketFreq", "Family"]
X = df.copy()
y = X.pop("Survived")

# Impute missing continuous values with the training median.
# Also add a boolean '_na' flag column so the model can learn
# that "this value was missing" is itself a signal worth keeping.
for col in cont_cols:
    if X[col].isnull().any():
        median_val = X[col].median()

        X[f"{col}_na"] = X[col].isnull().astype(bool)
        tst_df[f"{col}_na"] = tst_df[col].isnull().astype(bool)

        X[col] = X[col].fillna(median_val)
        tst_df[col] = tst_df[col].fillna(median_val)

# Impute missing categorical values with a "Missing" placeholder,
# then convert each category to an integer code.
# tst_df uses the same category set as X so codes stay consistent.
for col in cat_cols:
    X[col] = X[col].fillna("Missing").astype("category")

    tst_df[col] = pd.Categorical(
        tst_df[col].fillna("Missing"), categories=X[col].cat.categories
    )

    # +1 shifts codes so that 0 is a safe "unknown/padding" index,
    # which is the convention nn.Embedding expects.
    X[col] = X[col].cat.codes + 1
    tst_df[col] = tst_df[col].cat.codes + 1

# Z-Score normalization: subtract mean, divide by std.
# Keeps continuous features on a similar scale so no single
# feature dominates the gradient updates.
for col in cont_cols:
    mean_val = X[col].mean()
    std_val = X[col].std()

    if std_val != 0:
        X[col] = (X[col] - mean_val) / std_val
        tst_df[col] = (tst_df[col] - mean_val) / std_val


# Wrap data in a TensorDataset so DataLoader can batch it.
# Categorical features stay as long ints (needed for embedding lookup).
# Continuous features and labels become floats/longs respectively.
X_cat = torch.tensor(X[cat_cols].values, dtype=torch.long)
X_cont = torch.tensor(X[cont_cols].values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long)

dataset = TensorDataset(X_cat, X_cont, y_tensor)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Build embedding size tuples: (num_categories, embedding_dim).
# num_categories is max code + 1 to account for the +1 shift above.
# Embedding dim uses the fastai heuristic: min(50, (n + 5) // 2).
emb_szs = []
for col in cat_cols:
    num_cats = int(X[col].max() + 1)
    emb_dim = min(50, (num_cats + 5) // 2)
    emb_szs.append((num_cats, emb_dim))


### Neural Network ###
class Net(nn.Module):
    def __init__(self, emb_szs, n_cont):
        super().__init__()

        # One trainable embedding table per categorical column.
        # nn.Embedding(num_cats, dim) is a (num_cats × dim) weight matrix —
        # calling it with an integer index returns that row as a float vector.
        self.embeds = nn.ModuleList(
            [nn.Embedding(num_cats, dim) for num_cats, dim in emb_szs]
        )

        # The MLP input size = sum of all embedding dims + number of cont. features.
        n_embs = sum(dim for _, dim in emb_szs)
        total_input = n_embs + n_cont

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(total_input, 64),  # hidden layer 1
            nn.ReLU(),
            nn.Linear(64, 32),           # hidden layer 2
            nn.ReLU(),
            nn.Linear(32, 2),            # output: survived (0 or 1)
        )

    def forward(self, x_cat, x_cont):
        # x_cat: (batch, 5) — one integer code per categorical column per sample.
        # x_cat[:, i] slices column i → shape (batch,).
        # Passing that through embedding table e does a row-lookup:
        # each integer is replaced by its learned float vector → shape (batch, dim).
        embeddings = [e(x_cat[:, i]) for i, e in enumerate(self.embeds)]

        # Stack all embedding vectors side-by-side → (batch, sum of all dims)
        categorical_flat = torch.cat(embeddings, dim=1)

        # Append normalized continuous features → (batch, total_input)
        full_input = torch.cat([categorical_flat, x_cont], dim=1)

        logits = self.linear_relu_stack(full_input)
        return logits

model = Net(emb_szs=emb_szs, n_cont=len(cont_cols))
print(model)


### Training/Testing ###
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X_cat, X_cont, y) in enumerate(dataloader):
        
        # Forward pass
        pred = model(X_cat, X_cont)
        loss = loss_fn(pred, y)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # Update
        optimizer.step()

        # Print every 10 batches
        if batch % 10 == 0:
            loss_val, current = loss.item(), (batch + 1) * len(y)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X_cat, X_cont, y in dataloader:
            pred = model(X_cat, X_cont)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# 20 epochs - arbitrary, performed the same as 16 epochs on Kaggle
epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(train_loader, model, loss_fn)
print("Training complete")



### Save/Load ###
PATH = "./titanic_net.pth"
torch.save(model.state_dict(), PATH)
print(f"Saved PyTorch Model State to {PATH}")



### Kaggle Submission ###
model = Net(emb_szs=emb_szs, n_cont=len(cont_cols))
model.load_state_dict(torch.load(PATH, weights_only=True))

tst_cat = torch.tensor(tst_df[cat_cols].values, dtype=torch.long)
tst_cont = torch.tensor(tst_df[cont_cols].values, dtype=torch.float32)

model.eval()
with torch.no_grad():
    tst_logits = model(tst_cat, tst_cont)
    tst_preds = tst_logits.argmax(1).numpy() # pick class with highest logit

submission = pd.DataFrame({
    "PassengerId": passenger_ids.values,
    "Survived": tst_preds,
})
submission.to_csv("./submission.csv", index=False)
print("submission.csv created")
