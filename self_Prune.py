"""
Self-Pruning Neural Network (CIFAR-10)

Idea:
Each weight has a gate (0 to 1). During training,
model learns which weights are important and pushes others towards 0.

I used L1 penalty on gates to encourage sparsity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt



class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        # normal weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # gate scores (learnable)
        # shifted a bit negative so initial gates are small
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features) - 2)

    def forward(self, x):
        # sigmoid -> converts to range (0,1)
        # multiplied by 2 to make it sharper
        gates = torch.sigmoid(self.gate_scores * 2)

        # apply pruning (element-wise)
        pruned_weight = self.weight * gates

        return F.linear(x, pruned_weight, self.bias)



class PrunableNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = PrunableLinear(32 * 32 * 3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten image

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x



def sparsity_loss(model):
    """
    L1 penalty on gates.
    Using mean instead of sum so it doesn't blow up with model size.
    """
    loss = 0.0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores * 2)
            loss += torch.mean(gates)

    return loss


def train(model, train_loader, optimizer, lambda_sparse, device):
    model.train()
    total_loss = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        # normal classification loss
        cls_loss = F.cross_entropy(output, target)

        # sparsity loss
        sp_loss = sparsity_loss(model)

        # total loss
        loss = cls_loss + lambda_sparse * sp_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def test(model, test_loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            pred = output.argmax(dim=1)

            correct += (pred == target).sum().item()
            total += target.size(0)

    return correct / total


def compute_sparsity(model, threshold=1e-2):
    total = 0
    pruned = 0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores * 2)

            total += gates.numel()
            pruned += (gates < threshold).sum().item()

    return 100 * pruned / total


def plot_gates(model):
    all_gates = []

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores * 2).detach().cpu().numpy()
            all_gates.extend(gates.flatten())

    plt.hist(all_gates, bins=50)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate value")
    plt.ylabel("Count")
    plt.show()


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # CIFAR-10 preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # higher lambda since we are using mean
    lambdas = [0.1, 1, 10]

    results = []

    for lam in lambdas:
        print(f"\nTraining with lambda = {lam}")

        model = PrunableNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(10):
            loss = train(model, train_loader, optimizer, lam, device)
            print(f"Epoch {epoch+1} | Loss = {loss:.4f}")

        acc = test(model, test_loader, device)
        sparsity = compute_sparsity(model)

        print(f"Accuracy: {acc:.4f}")
        print(f"Sparsity: {sparsity:.2f}%")

        # debug (optional but useful)
        g = torch.sigmoid(model.fc1.gate_scores * 2)
        print("Gate stats -> min:", g.min().item(),
              "mean:", g.mean().item(),
              "max:", g.max().item())

        results.append((lam, acc, sparsity))

        # plot only once (last run)
        plot_gates(model)

    # final results
    print("\nFinal Results:")
    print("Lambda\tAccuracy\tSparsity")
    for lam, acc, sp in results:
        print(f"{lam}\t{acc:.4f}\t\t{sp:.2f}")
