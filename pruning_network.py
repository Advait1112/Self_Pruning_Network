import torch
import torch.nn as torch_nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

batch_size = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

print(f"Training batches: {len(trainloader)}, Testing batches: {len(testloader)}")

class PrunableLinear(torch_nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = torch_nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = torch_nn.Parameter(torch.empty(out_features))
        
        self.gate_scores = torch_nn.Parameter(torch.empty((out_features, in_features)))
        
        self.reset_parameters()

    def reset_parameters(self):
        torch_nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = torch_nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch_nn.init.uniform_(self.bias, -bound, bound)

        torch_nn.init.normal_(self.gate_scores, mean=3.0, std=0.1)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        
        pruned_weights = self.weight * gates
        
        return F.linear(x, pruned_weights, self.bias)

class SelfPruningNet(torch_nn.Module):
    def __init__(self):
        super(SelfPruningNet, self).__init__()
        self.flatten = torch_nn.Flatten()
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10) # 10 output classes

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def get_sparsity_loss(model):
    """Calculates the normalized L1 norm of all gate values."""
    l1_loss = 0.0
    total_gates = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            l1_loss += torch.sum(gates)
            total_gates += gates.numel()
            
    return l1_loss / total_gates if total_gates > 0 else 0.0

def train_and_evaluate(lmbda=0.0001, epochs=5, threshold=1e-2):
    print(f"\n--- Training with Lambda = {lmbda} ---")
    model = SelfPruningNet().to(device)
    criterion = torch_nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            cls_loss = criterion(outputs, labels)
            sparsity_loss = get_sparsity_loss(model)
            total_loss = cls_loss + (lmbda * sparsity_loss)
            
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {running_loss/len(trainloader):.4f}")

    model.eval()
    correct = 0
    total = 0
    total_weights = 0
    pruned_weights_count = 0
    all_gates = []

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        accuracy = 100 * correct / total

        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores).cpu().view(-1)
                all_gates.extend(gates.numpy())
                
                total_weights += gates.numel()
                pruned_weights_count += torch.sum(gates < threshold).item()
                
        sparsity_level = 100 * (pruned_weights_count / total_weights)

    print(f"Test Accuracy: {accuracy:.2f}% | Sparsity Level: {sparsity_level:.2f}%")
    return accuracy, sparsity_level, all_gates, model

lambda_values = [0.0, 10.0, 50.0, 100.0] 
results = []
best_gates = None
highest_sparsity = -1.0 # Start below 0

print("Lambda | Test Accuracy | Sparsity Level (%)")
print("________________________________________________")

for l in lambda_values:
    acc, sparsity, gates, _ = train_and_evaluate(lmbda=l, epochs=15)
    results.append((l, acc, sparsity))
    
    if sparsity >= highest_sparsity:
        highest_sparsity = sparsity
        best_gates = gates

print("\n--- Final Results Summary ---")
print("Lambda\t| Accuracy\t| Sparsity (%)")
print("_______________________________________")
for l, acc, sp in results:
    print(f"{l}\t| {acc:.2f}%\t| {sp:.2f}%")

plt.figure(figsize=(10, 6))
plt.hist(best_gates, bins=50, color='blue', alpha=0.7)
plt.title('Distribution of Final Gate Values')
plt.xlabel('Gate Value (after Sigmoid)')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.yscale('log')
plt.show()