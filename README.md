# Shallow Universal Polynomial Networks

[![Python Tests](https://github.com/sandialabs/SUPN/actions/workflows/python-tests.yml/badge.svg)](https://github.com/sandialabs/SUPN/actions/workflows/python-tests.yml)
[![Coverage](https://github.com/sandialabs/SUPN/blob/gh-pages/badges/coverage.svg)](https://github.com/sandialabs/SUPN/actions)
[![TMLR](https://img.shields.io/badge/TMLR-published-brightgreen)](https://openreview.net/forum?id=srHGkCrSTM)

SUPNs (Shallow Universal Polynomial Networks) are parsimonious, yet highly expressive, surrogate models of the form

```math
\hat{f}(x) = \sum_{n=1}^N w_n \tanh\left( \sum_{m=0}^M a_{n,m} \, T_m(x) \right) \, ,
```

where $`T_m`$ is the Chebyshev polynomial of degree $m$. SUPNs have been shown to achieve equal or better accuracy to DNNs, KANs, and even polynomials (on nonsmooth problems) with a similar parameter count \[1\]. They also come equipped with convergence theory that goes beyond the usual universal approximation theorems. Single-neuron SUPNs have accuracy bounded by the best polynomial approximation of a given degree $M$ \[1\]. Multi-neuron SUPNs are capable of approximating a spline interpolant with accuracy explicitly in terms of basis size, spline smoothness, and $M$ \[2\].

## Installation
```bash
git clone <this-repo>
cd SUPN
pip install -e .
```

## Usage
See the [`tutorials`](./tutorials/) folder.

## Dependencies

- Python (>= 3.11 supported)
- PyTorch

# Acknowledgements

Development of this software was funded by the Laboratory-Directed Research and Development (LDRD) program at Sandia National Laboratories.

## Citations

If you use this code in your research, please use the following citations:

1. Z. Morrow, M. Penwarden, B. Chen, A. Javeed, A. Narayan, and J. D. Jakeman. "SUPN: Shallow Universal Polynomial Networks for Efficient Function Approximation." _Transactions on Machine Learning Research_ (2026). URL: https://openreview.net/forum?id=srHGkCrSTM.

2. Z. Morrow, Y. Fu, P. Roy, and M. Penwarden. "Physics-Informed Shallow Universal Polynomial Networks for High-Frequency and High-Order Systems." In preparation (2026).

BibTeX:

```bibtex
@article{morrow2026supn,
    title = {{SUPN}: Shallow Universal Polynomial Networks for Efficient Function Approximation},
    author = {Zachary Morrow and Michael Penwarden and Brian Chen and Aurya Javeed and Akil Narayan and John Davis Jakeman},
    journal = {Transactions on Machine Learning Research},
    issn = {2835-8856},
    year = {2026},
    url = {https://openreview.net/forum?id=srHGkCrSTM}
}
```

