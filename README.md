# Modular Form Activation

This project explores using unusual activation functions in neural networks, specifically those derived from modular forms. In particular, we focus on the **Dedekind eta function** as an activation function.

## Dedekind Eta Function as Activation Function

The Dedekind eta function is a modular form defined on the complex upper half-plane with significant applications in number theory and theoretical physics. It is defined as:

$$
\eta(\tau) = e^{\pi i \tau / 12} \prod_{n=1}^\infty \left( 1 - e^{2\pi i n \tau} \right)
$$

where $\tau$ is a complex number with a positive imaginary part.

In this project, we implement the Dedekind eta function using Euler's formula and integrate it as an activation function within neural network architectures.

### Implementation Details

We approximate the infinite product in the Dedekind eta function using a finite sum based on Euler's formula:

$$
\eta(\tau) \approx \sum_{n=-N}^{N} e^{\pi i n} \, e^{3 \pi i \left( n + \tfrac{1}{6} \right)^2 \tau}
$$

where $N$ is the number of terms used in the approximation.

The implementation can be found in the `DedekindEta` class in the `__init__.py` file.

