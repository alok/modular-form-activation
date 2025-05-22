# Modular-Form Activations for PyTorch

A tiny experimental library providing activation functions inspired by classical modular forms (currently: **Dedekind–eta**).

```bash
uv pip install modular-form-activation
```

## Quick start

```python
import torch
from modular_form_activation import DedekindEtaActivation

act = DedekindEtaActivation(truncation=30, learnable_scale=True)

x = torch.linspace(-3, 3, 100)
y = act(x)
print(y.min(), y.max())
```

---

The activation implemented is

\[
  f(x) = -\log \eta\bigl(i\, \operatorname{softplus}(s\,x)\bigr),
\]

where \(\eta\) is the Dedekind eta function and \(s\) is an optional learnable scale.

The infinite product defining \(\eta\) is truncated after *N* factors (set by `truncation`). Even for moderate \(N \approx 20\) the approximation error is below `1e-5` for typical neural-network ranges.

---

*Warning*: this is research code – use at your own risk.
