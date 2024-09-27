#! /usr/bin/env python3
# %%
from __future__ import annotations
import cmath
from collections import defaultdict
from dataclasses import dataclass
from numbers import Number
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
import math

from torch import vmap  # Import vmap from torch.func


@dataclass
class Config:
    num_terms: int = 10
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001



def root_of_unity(n: Number) -> torch.Tensor:
    n = torch.tensor(n,dtype=torch.cfloat)
    return torch.exp(1j * 2*math.pi * n)

class DedekindEta(nn.Module):
    """
    PyTorch module to compute the Dedekind Eta function using Euler's formula.

    For any complex number τ with Im(τ) > 0, the Dedekind Eta function is defined as:
    η(τ) = ∑_{n=-∞}^{∞} e^{πi n} e^{3πi (n + 1⁄6)² τ}.

    Args:
        config (Config): Configuration for the number of Fourier terms.
    """

    def __init__(self, config: Config=Config(num_terms=100)) -> None:
        super().__init__()
        self.config = config
        self.register_buffer(
            "n",
            torch.arange(
                -self.config.num_terms,
                self.config.num_terms + 1,
                dtype=torch.float32,
            )
        )
        self.register_buffer(
            "exponent_base",
            3 * math.pi * 1j * (self.n + 1 / 6) ** 2,  # Shape: [1, 1, n_terms]
        )
        self.register_buffer(
            "root_of_unity", torch.exp(1j * math.pi * self.n)  # Shape: [1, 1, n_terms]
        )

    def forward(
        self, tau: Float[torch.Tensor, ""] | Number
    ) -> Float[torch.Tensor, ""]:
        """
        Forward pass to compute the Dedekind Eta function using Euler's formula.

        Args:
            tau (torch.Tensor): Input tensor with shape [batch, features] and Im(tau) > 0.
                               It should be a complex tensor (dtype=torch.cfloat).

        Returns:
            torch.Tensor: Computed Dedekind Eta values with shape [batch, features].
        """
        if isinstance(tau, Number):
            tau = torch.tensor(tau,dtype=torch.cfloat)
        if not torch.is_complex(tau):
            tau = tau.to(torch.cfloat)

        exponent_tau = self.exponent_base * tau.unsqueeze(-1)  # [features, n_terms]
        total_terms = self.root_of_unity * torch.exp(exponent_tau)  # [features, n_terms]
        eta = total_terms.sum(dim=-1)  # Sum over n_terms dimension

        return eta


class DedekindEtaSpecialValues(nn.Module):
    """
    PyTorch module to compute special values of the Dedekind Eta function.
    """

    def __init__(self):
        super().__init__()

    def eta_i(self):
        return math.gamma(1 / 4) / (2 * math.pi ** (3 / 4))

    def eta_half_i(self):
        return math.gamma(1 / 4) / (2 ** (7 / 8) * math.pi ** (3 / 4))

    def eta_2i(self):
        return math.gamma(1 / 4) / (2 ** (11 / 8) * math.pi ** (3 / 4))

    def eta_3i(self):
        return math.gamma(1 / 4) / (
            2
            * 3 ** (1 / 3)
            * (3 + 2 * torch.sqrt(torch.tensor(3.0))) ** (1 / 12)
            * math.pi ** (3 / 4)
        )

    def eta_4i(self):
        return ((-1 + torch.sqrt(torch.tensor(2.0))) ** (1 / 4) * math.gamma(1 / 4)) / (
            2 ** (29 / 16) * math.pi ** (3 / 4)
        )

    def eta_e_2pi_i_3(self):
        return (
            torch.exp(torch.tensor(-math.pi * 1j / 24))
            * (3 ** (1 / 8) * math.gamma(1 / 3) ** (3 / 2))
            / (2 * math.pi)
        )

    def eta_single_value(self, _):
        """
        Returns a single special value, ignoring input.

        Args:
            _: Dummy input.

        Returns:
            torch.Tensor: A single special value.
        """
        value = self.eta_i()
        return torch.tensor(value)

    def forward(self, tau: torch.Tensor):
        """
        Forward pass to compute special values of the Dedekind Eta function.

        Args:
            tau (torch.Tensor): Input tensor with special values [i, 0.5i, 2i, 3i, 4i, e^(2πi/3)].
                                 (Batch size is irrelevant here as special values are fixed.)

        Returns:
            torch.Tensor: Computed special Dedekind Eta values.
        """
        special_values = torch.tensor(
            [
                self.eta_i(),
                self.eta_half_i(),
                self.eta_2i(),
                self.eta_3i(),
                self.eta_4i(),
                self.eta_e_2pi_i_3(),
            ]
        )
        return special_values


def test_dedekind_eta():
    config = Config(num_terms=1000)  # Large number of terms for accuracy
    dedekind_eta = DedekindEta(config).eval()
    special_values_module = DedekindEtaSpecialValues()

    # Test inputs
    tau = torch.tensor(
        [1j, 0.5j, 2j, 3j, 4j, torch.exp(torch.tensor(2j * math.pi / 3))]
    )

    # Compute values using DedekindEta
    computed_values = dedekind_eta(tau)

    # Compute special values
    expected_values = special_values_module(tau)
    print(f"{expected_values = }")
    print(f"{computed_values = }")
    # Compare results
    torch.testing.assert_close(computed_values, expected_values, rtol=1e-4, atol=1e-4)
    print("DedekindEta implementation matches special values within tolerance.")


# Run the test
test_dedekind_eta()


# %%
class JInvariant(nn.Module):
    """
    PyTorch module to compute the j-invariant using the Dedekind Eta function.

    The j-invariant is computed using the formula:
    j(τ) = ((η(2τ)/η(τ))^8 + 2^8 * (η(τ)/η(2τ))^16)^3

    Args:
        config (Config): Configuration for the number of Fourier terms.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config



    def forward(
        self, tau: Float[torch.Tensor, "features"]
    ) -> Float[torch.Tensor, "features"]:
        η = DedekindEta(self.config)
        eta_tau = η(tau)
        eta_2tau = η(2 * tau)
        
        ratio1 = (eta_2tau / eta_tau) ** 8
        ratio2 = (eta_tau / eta_2tau) ** 16
        
        j = (ratio1 + 256 * ratio2) ** 3
        return j


j = JInvariant(config=Config(num_terms=100))
j((1+cmath.sqrt(-163))/2)
assert j(1j) == 12**3
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
        self.coeffs = nn.Parameter(
            torch.randn(self.num_terms, dtype=torch.cfloat) * 0.1
        )
        self.n = torch.arange(
            1, self.num_terms + 1, dtype=torch.float32
        ).reshape(1, -1)  # Shape: [1, num_terms]

    def activation_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute activation for a single input x.

        Args:
            x (torch.Tensor): Single input tensor.

        Returns:
            torch.Tensor: Activated value.
        """
        exponent = 1j * self.n * x  # [1, num_terms]
        y = self.coeffs * torch.exp(exponent)  # [num_terms]
        return y.sum().real  # Returning the real part

    def forward(
        self, x: Float[torch.Tensor, "batch features"]
    ) -> Float[torch.Tensor, "batch features"]:
        """
        Forward pass of the activation function using complex exponentials.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activated tensor.
        """
        return vmap(self.activation_single)(x)


# Inline tests for ModularFormActivation and DedekindEta
class TestModularFormActivation(unittest.TestCase):
    def test_output_shape(self):
        activation = ModularFormActivation(config=Config(num_terms=5))
        input_tensor = torch.randn(10, 20)
        output = activation(input_tensor)
        self.assertEqual(output.shape, (10, 20))

    def test_dedekind_eta_known_value(self):
        config = Config(num_terms=1)
        dedekind_eta = DedekindEta(config=config)
        input_z = torch.tensor([0.0], dtype=torch.float32)
        output = dedekind_eta(input_z)
        expected = (
            dedekind_eta.coeffs[0].real + dedekind_eta.coeffs[0].imag
        )  # Since exp(0) = 1 + 0j
        self.assertAlmostEqual(
            output.item(),
            (dedekind_eta.coeffs[0] * torch.exp(1j * 1 * input_z)).real.item(),
            places=5,
        )

    def test_j_invariant_known_value(self):
        config = Config(num_terms=1)
        j_invariant = JInvariant(config=config)
        input_z = torch.tensor([0.0], dtype=torch.float32)
        output = j_invariant(input_z)
        expected = (
            j_invariant.coeffs[0].real + j_invariant.coeffs[0].imag
        )  # Simplified for n=1 and z=0
        self.assertAlmostEqual(
            output.item(),
            (j_invariant.coeffs[0] * torch.exp(2j * 1 * input_z)).real.item(),
            places=5,
        )


# Visualization of the activation function
def plot_activation(
    activation_fn: Callable[[torch.Tensor], torch.Tensor], title: str
) -> None:
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
    fig.add_trace(go.Scatter(x=x.numpy(), y=y.numpy(), mode="lines", name=title))
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

    def forward(
        self, x: Float[torch.Tensor, "batch 1 28 28"]
    ) -> Float[torch.Tensor, "batch 10"]:
        x = self.flatten(x)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x


# Training function
def train_model(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> Tuple[list, list]:
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
def evaluate_model(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[list, list]:
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
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Initialize activations
    config = Config(num_terms=10)
    modular_activation = ModularFormActivation(config=config).to(device)
    dedekind_eta = DedekindEta(config=config).to(device)
    j_invariant = JInvariant(config=config).to(device)
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
        loss_m, acc_m = train_model(
            model_modular, device, train_loader, optimizer_modular, criterion
        )
        loss_d, acc_d = train_model(
            model_dedekind, device, train_loader, optimizer_dedekind, criterion
        )
        loss_j, acc_j = train_model(
            model_j, device, train_loader, optimizer_j, criterion
        )
        loss_g, acc_g = train_model(
            model_gelu, device, train_loader, optimizer_gelu, criterion
        )

        history["modular_loss"].append(np.mean(loss_m))
        history["modular_acc"].append(np.mean(acc_m))
        history["dedekind_loss"].append(np.mean(loss_d))
        history["dedekind_acc"].append(np.mean(acc_d))
        history["j_invariant_loss"].append(np.mean(loss_j))
        history["j_invariant_acc"].append(np.mean(acc_j))
        history["gelu_loss"].append(np.mean(loss_g))
        history["gelu_acc"].append(np.mean(acc_g))

        print(f"Epoch {epoch+1}/{epochs}")
        print(
            f"Modular Activation - Loss: {history['modular_loss'][-1]:.4f}, Accuracy: {history['modular_acc'][-1]*100:.2f}%"
        )
        print(
            f"Dedekind Eta - Loss: {history['dedekind_loss'][-1]:.4f}, Accuracy: {history['dedekind_acc'][-1]*100:.2f}%"
        )
        print(
            f"J-Invariant - Loss: {history['j_invariant_loss'][-1]:.4f}, Accuracy: {history['j_invariant_acc'][-1]*100:.2f}%"
        )
        print(
            f"GELU Activation - Loss: {history['gelu_loss'][-1]:.4f}, Accuracy: {history['gelu_acc'][-1]*100:.2f}%"
        )

    # Plot training loss
    fig_loss = go.Figure()
    fig_loss.add_trace(
        go.Scatter(
            x=list(range(1, epochs + 1)),
            y=history["modular_loss"],
            mode="lines+markers",
            name="Modular Activation",
        )
    )
    fig_loss.add_trace(
        go.Scatter(
            x=list(range(1, epochs + 1)),
            y=history["dedekind_loss"],
            mode="lines+markers",
            name="Dedekind Eta",
        )
    )
    fig_loss.add_trace(
        go.Scatter(
            x=list(range(1, epochs + 1)),
            y=history["j_invariant_loss"],
            mode="lines+markers",
            name="J-Invariant",
        )
    )
    fig_loss.add_trace(
        go.Scatter(
            x=list(range(1, epochs + 1)),
            y=history["gelu_loss"],
            mode="lines+markers",
            name="GELU Activation",
        )
    )
    fig_loss.update_layout(
        title="Training Loss Comparison", xaxis_title="Epoch", yaxis_title="Loss"
    )
    fig_loss.show()

    # Plot training accuracy
    fig_acc = go.Figure()
    fig_acc.add_trace(
        go.Scatter(
            x=list(range(1, epochs + 1)),
            y=[acc * 100 for acc in history["modular_acc"]],
            mode="lines+markers",
            name="Modular Activation",
        )
    )
    fig_acc.add_trace(
        go.Scatter(
            x=list(range(1, epochs + 1)),
            y=[acc * 100 for acc in history["dedekind_acc"]],
            mode="lines+markers",
            name="Dedekind Eta",
        )
    )
    fig_acc.add_trace(
        go.Scatter(
            x=list(range(1, epochs + 1)),
            y=[acc * 100 for acc in history["j_invariant_acc"]],
            mode="lines+markers",
            name="J-Invariant",
        )
    )
    fig_acc.add_trace(
        go.Scatter(
            x=list(range(1, epochs + 1)),
            y=[acc * 100 for acc in history["gelu_acc"]],
            mode="lines+markers",
            name="GELU Activation",
        )
    )
    fig_acc.update_layout(
        title="Training Accuracy Comparison",
        xaxis_title="Epoch",
        yaxis_title="Accuracy (%)",
    )
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
    unittest.main(argv=[""], verbosity=2, exit=False)


# Execute when running the script
plot_activation(
    ModularFormActivation(config=Config(num_terms=10)), "Modular Form Activation"
)
plot_activation(
    DedekindEta(config=Config(num_terms=10)), "Dedekind Eta Activation"
)
plot_activation(
    JInvariant(config=Config(num_terms=10)), "J-Invariant Activation"
)
plot_activation(nn.GELU(), "GELU Activation")
run_tests()
compare_activations()


def main(): ...


if __name__ == "__main__":
    main()


# %%
