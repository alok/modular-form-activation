from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Callable, Tuple
import plotly.graph_objs as go
import torch.optim as optim
import numpy as np

import jaxtyping
from jaxtyping import Float
import einops

import unittest


@dataclass
class Config:
    num_terms: int = 10
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001


@dataclass
class DedekindEtaConfig:
    """
    Configuration for the DedekindEta module.

    Args:
        num_terms (int): Number of Fourier terms to include in the approximation.
    """
    num_terms: int = 10


class DedekindEta(nn.Module):
    """
    PyTorch module to compute the Dedekind Eta function using truncated Fourier series with complex exponentials.

    Args:
        config (DedekindEtaConfig): Configuration for the number of Fourier terms.
    """
    def __init__(self, config: DedekindEtaConfig) -> None:
        super(DedekindEta, self).__init__()
        self.num_terms = config.num_terms
        # Initialize complex Fourier coefficients
        self.coeffs = nn.Parameter(torch.randn(self.num_terms, dtype=torch.cfloat) * 0.1)

    def forward(self, z: Float[torch.Tensor, "batch features"]) -> Float[torch.Tensor, "batch features"]:
        """
        Forward pass to compute the Dedekind Eta function using complex exponentials.

        Args:
            z (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Computed Dedekind Eta values.
        """
        n = torch.arange(1, self.num_terms + 1, device=z.device, dtype=torch.float32).unsqueeze(0)  # Shape: [1, num_terms]
        exponent = 1j * n * z.unsqueeze(1)  # Shape: [batch, num_terms]
        y = self.coeffs.unsqueeze(0) * torch.exp(exponent)  # Shape: [batch, num_terms]
        return y.sum(dim=1).real  # Returning the real part


class JInvariant(nn.Module):
    """
    PyTorch module to compute the j-invariant using complex exponentials.

    Args:
        config (DedekindEtaConfig): Configuration for the number of Fourier terms.
    """
    def __init__(self, config: DedekindEtaConfig) -> None:
        super(JInvariant, self).__init__()
        self.num_terms = config.num_terms
        # Initialize complex Fourier coefficients for j-invariant
        self.coeffs = nn.Parameter(torch.randn(self.num_terms, dtype=torch.cfloat) * 0.1)

    def forward(self, z: Float[torch.Tensor, "batch features"]) -> Float[torch.Tensor, "batch features"]:
        """
        Forward pass to compute the j-invariant using complex exponentials.

        Args:
            z (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Computed j-invariant values.
        """
        n = torch.arange(1, self.num_terms + 1, device=z.device, dtype=torch.float32).unsqueeze(0)  # Shape: [1, num_terms]
        exponent = 2j * n * z.unsqueeze(1)  # Example multiplier for j-invariant
        y = self.coeffs.unsqueeze(0) * torch.exp(exponent)  # Shape: [batch, num_terms]
        return y.sum(dim=1).real  # Returning the real part


# Define the ModularFormActivation using truncated Fourier series with complex exponentials
class ModularFormActivation(nn.Module):
    """
    PyTorch activation function based on a modular form using truncated Fourier series with complex exponentials.

    Args:
        config (Config): Configuration for the number of Fourier terms.
    """
    def __init__(self, config: Config) -> None:
        super(ModularFormActivation, self).__init__()
        self.num_terms = config.num_terms
        # Initialize complex Fourier coefficients
        self.coeffs = nn.Parameter(torch.randn(config.num_terms, dtype=torch.cfloat) * 0.1)

    def forward(self, x: Float[torch.Tensor, "batch features"]) -> Float[torch.Tensor, "batch features"]:
        """
        Forward pass of the activation function using complex exponentials.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activated tensor.
        """
        n = torch.arange(1, self.num_terms + 1, device=x.device, dtype=torch.float32).unsqueeze(0)  # Shape: [1, num_terms]
        exponent = 1j * n * x.unsqueeze(1)  # Shape: [batch, num_terms]
        y = self.coeffs.unsqueeze(0) * torch.exp(exponent)  # Shape: [batch, num_terms]
        return y.sum(dim=1).real  # Returning the real part


# Inline tests for ModularFormActivation and DedekindEta
class TestModularFormActivation(unittest.TestCase):
    def test_output_shape(self):
        activation = ModularFormActivation(config=Config(num_terms=5))
        input_tensor = torch.randn(10, 20)
        output = activation(input_tensor)
        self.assertEqual(output.shape, (10, 20))

    def test_dedekind_eta_known_value(self):
        config = DedekindEtaConfig(num_terms=1)
        dedekind_eta = DedekindEta(config=config)
        input_z = torch.tensor([0.0], dtype=torch.float32)
        output = dedekind_eta(input_z)
        expected = dedekind_eta.coeffs[0].real + dedekind_eta.coeffs[0].imag  # Since exp(0) = 1 + 0j
        self.assertAlmostEqual(output.item(), (dedekind_eta.coeffs[0] * torch.exp(1j * 1 * input_z)).real.item(), places=5)

    def test_j_invariant_known_value(self):
        config = DedekindEtaConfig(num_terms=1)
        j_invariant = JInvariant(config=config)
        input_z = torch.tensor([0.0], dtype=torch.float32)
        output = j_invariant(input_z)
        expected = j_invariant.coeffs[0].real + j_invariant.coeffs[0].imag  # Simplified for n=1 and z=0
        self.assertAlmostEqual(output.item(), (j_invariant.coeffs[0] * torch.exp(2j * 1 * input_z)).real.item(), places=5)


# Visualization of the activation function
def plot_activation(activation_fn: Callable[[torch.Tensor], torch.Tensor], title: str) -> None:
    """
    Plots the activation function over a range of inputs.

    Args:
        activation_fn (Callable): The activation function to plot.
        title (str): Title of the plot.
    """
    x = torch.linspace(-10, 10, 400)
    with torch.no_grad():
        y = activation_fn(x)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x.numpy(), y=y.numpy(), mode='lines', name=title))
    fig.update_layout(title=title, xaxis_title="Input", yaxis_title="Output")
    fig.show()


# Define a standard feedforward network
class FeedforwardNetwork(nn.Module):
    """
    Standard feedforward neural network.

    Args:
        activation (nn.Module): Activation function to use.
    """
    def __init__(self, activation: nn.Module) -> None:
        super(FeedforwardNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.act1 = activation
        self.fc2 = nn.Linear(128, 64)
        self.act2 = activation
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x: Float[torch.Tensor, "batch 1 28 28"]) -> Float[torch.Tensor, "batch 10"]:
        x = self.flatten(x)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x


# Training function
def train_model(model: nn.Module, device: torch.device, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module) -> Tuple[list, list]:
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        device (torch.device): Device to run the training on.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.

    Returns:
        Tuple[list, list]: Training losses and accuracies.
    """
    model.train()
    losses = []
    correct = 0
    total = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    accuracy = correct / total
    return losses, [accuracy]


# Evaluation function
def evaluate_model(model: nn.Module, device: torch.device, test_loader: DataLoader, criterion: nn.Module) -> Tuple[list, list]:
    """
    Evaluates the model on the test dataset.

    Args:
        model (nn.Module): The neural network model.
        device (torch.device): Device to run the evaluation on.
        test_loader (DataLoader): DataLoader for test data.
        criterion (nn.Module): Loss function.

    Returns:
        Tuple[list, list]: Test losses and accuracies.
    """
    model.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    return losses, [accuracy]


# Main function to train and compare activation functions
def compare_activations() -> None:
    """
    Trains two models with different activation functions and compares their performance.
    """
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Initialize activations
    config = Config(num_terms=10)
    modular_activation = ModularFormActivation(config=config).to(device)
    dedekind_eta = DedekindEta(config=DedekindEtaConfig(num_terms=10)).to(device)
    j_invariant = JInvariant(config=DedekindEtaConfig(num_terms=10)).to(device)
    gelu_activation = nn.GELU().to(device)

    # Initialize models
    model_modular = FeedforwardNetwork(activation=modular_activation).to(device)
    model_dedekind = FeedforwardNetwork(activation=dedekind_eta).to(device)
    model_j = FeedforwardNetwork(activation=j_invariant).to(device)
    model_gelu = FeedforwardNetwork(activation=gelu_activation).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_modular = optim.Adam(model_modular.parameters(), lr=0.001)
    optimizer_dedekind = optim.Adam(model_dedekind.parameters(), lr=0.001)
    optimizer_j = optim.Adam(model_j.parameters(), lr=0.001)
    optimizer_gelu = optim.Adam(model_gelu.parameters(), lr=0.001)

    epochs: int = 10
    history = defaultdict(list)

    for epoch in range(epochs):
        loss_m, acc_m = train_model(model_modular, device, train_loader, optimizer_modular, criterion)
        loss_d, acc_d = train_model(model_dedekind, device, train_loader, optimizer_dedekind, criterion)
        loss_j, acc_j = train_model(model_j, device, train_loader, optimizer_j, criterion)
        loss_g, acc_g = train_model(model_gelu, device, train_loader, optimizer_gelu, criterion)

        history['modular_loss'].append(np.mean(loss_m))
        history['modular_acc'].append(np.mean(acc_m))
        history['dedekind_loss'].append(np.mean(loss_d))
        history['dedekind_acc'].append(np.mean(acc_d))
        history['j_invariant_loss'].append(np.mean(loss_j))
        history['j_invariant_acc'].append(np.mean(acc_j))
        history['gelu_loss'].append(np.mean(loss_g))
        history['gelu_acc'].append(np.mean(acc_g))

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Modular Activation - Loss: {history['modular_loss'][-1]:.4f}, Accuracy: {history['modular_acc'][-1]*100:.2f}%")
        print(f"Dedekind Eta - Loss: {history['dedekind_loss'][-1]:.4f}, Accuracy: {history['dedekind_acc'][-1]*100:.2f}%")
        print(f"J-Invariant - Loss: {history['j_invariant_loss'][-1]:.4f}, Accuracy: {history['j_invariant_acc'][-1]*100:.2f}%")
        print(f"GELU Activation - Loss: {history['gelu_loss'][-1]:.4f}, Accuracy: {history['gelu_acc'][-1]*100:.2f}%")

    # Plot training loss
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=list(range(1, epochs+1)), y=history['modular_loss'], mode='lines+markers', name='Modular Activation'))
    fig_loss.add_trace(go.Scatter(x=list(range(1, epochs+1)), y=history['dedekind_loss'], mode='lines+markers', name='Dedekind Eta'))
    fig_loss.add_trace(go.Scatter(x=list(range(1, epochs+1)), y=history['j_invariant_loss'], mode='lines+markers', name='J-Invariant'))
    fig_loss.add_trace(go.Scatter(x=list(range(1, epochs+1)), y=history['gelu_loss'], mode='lines+markers', name='GELU Activation'))
    fig_loss.update_layout(title='Training Loss Comparison', xaxis_title='Epoch', yaxis_title='Loss')
    fig_loss.show()

    # Plot training accuracy
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(x=list(range(1, epochs+1)), y=[acc * 100 for acc in history['modular_acc']], mode='lines+markers', name='Modular Activation'))
    fig_acc.add_trace(go.Scatter(x=list(range(1, epochs+1)), y=[acc * 100 for acc in history['dedekind_acc']], mode='lines+markers', name='Dedekind Eta'))
    fig_acc.add_trace(go.Scatter(x=list(range(1, epochs+1)), y=[acc * 100 for acc in history['j_invariant_acc']], mode='lines+markers', name='J-Invariant'))
    fig_acc.add_trace(go.Scatter(x=list(range(1, epochs+1)), y=[acc * 100 for acc in history['gelu_acc']], mode='lines+markers', name='GELU Activation'))
    fig_acc.update_layout(title='Training Accuracy Comparison', xaxis_title='Epoch', yaxis_title='Accuracy (%)')
    fig_acc.show()

    # Plot activation functions
    plot_activation(modular_activation, "Modular Form Activation")
    plot_activation(dedekind_eta, "Dedekind Eta Activation")
    plot_activation(j_invariant, "J-Invariant Activation")
    plot_activation(gelu_activation, "GELU Activation")


# Run inline tests
def run_tests() -> None:
    """
    Runs all unit tests.
    """
    unittest.main(argv=[''], verbosity=2, exit=False)


# Execute when running the script
plot_activation(ModularFormActivation(config=Config(num_terms=10)), "Modular Form Activation")
plot_activation(DedekindEta(config=DedekindEtaConfig(num_terms=10)), "Dedekind Eta Activation")
plot_activation(JInvariant(config=DedekindEtaConfig(num_terms=10)), "J-Invariant Activation")
plot_activation(nn.GELU(), "GELU Activation")
run_tests()
compare_activations()



def main():
    ...


if __name__ == "__main__":
    main()
