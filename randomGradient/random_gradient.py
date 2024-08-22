import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x


class RandomGradientTrainer:
    def __init__(self, model, loss_fn, num_gradients=10, learning_rate=0.01):
        self.model = model
        self.loss_fn = loss_fn
        self.num_gradients = num_gradients
        self.learning_rate = learning_rate

    def generate_random_gradients(self):
        gradients = []
        for _ in range(self.num_gradients):
            grad = {}
            for name, param in self.model.named_parameters():
                grad[name] = torch.randn_like(param.data)
            gradients.append(grad)
        return gradients

    def apply_gradient(self, gradient):
        for name, param in self.model.named_parameters():
            param.data -= self.learning_rate * gradient[name]

    def train_step(self, inputs, targets):
        original_state = {name: param.data.clone() for name, param in self.model.named_parameters()}
        original_loss = self.loss_fn(self.model(inputs), targets)

        best_loss = float('inf')
        best_gradient = None

        gradients = self.generate_random_gradients()

        for gradient in gradients:
            self.apply_gradient(gradient)
            loss = self.loss_fn(self.model(inputs), targets)

            if loss < best_loss:
                best_loss = loss
                best_gradient = gradient

            # Reset model to original state
            for name, param in self.model.named_parameters():
                param.data = original_state[name].clone()

        if best_gradient is not None and best_loss < original_loss:
            self.apply_gradient(best_gradient)
            return best_loss
        else:
            return original_loss

    def train(self, train_loader, num_epochs):
        for epoch in range(num_epochs):
            total_loss = 0
            for inputs, targets in train_loader:
                loss = self.train_step(inputs, targets)
                total_loss += loss

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


# Generate some dummy data
np.random.seed(42)
X = np.random.rand(1000, 10)
y = np.sum(X, axis=1) > 5  # Binary classification task

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).unsqueeze(1)

# Create DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the model and trainer
model = SimpleNN(input_size=10, hidden_size=20, output_size=1)
loss_fn = nn.BCEWithLogitsLoss()
trainer = RandomGradientTrainer(model, loss_fn, num_gradients=10    , learning_rate=0.02)

# Train the model
trainer.train(train_loader, num_epochs=50)

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, targets in train_loader:
        outputs = model(inputs)
        predicted = (outputs > 0).float()
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")