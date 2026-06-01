# Uncertainty & Bayesian

Deterministic cores for uncertainty-aware models. Each is a plain `equinox.Module` with
no probabilistic-framework dependency; downstream libraries wrap them to add priors or
posteriors.

## Random features

Random Fourier features approximate a stationary kernel,
$\phi(x) = \sqrt{2/D}\,\cos(Wx + b)$, with optional orthogonal projections for lower
variance.

::: geonnax.randfeat

## Gaussian-process basis maps

Spectral-normalised GP (SNGP) and variational sparse-spectrum GP (VSSGP) cores that turn
a network's last layer into a GP via a random-feature basis and a Laplace covariance.

::: geonnax.sngp

::: geonnax.vssgp

## Deep ensembles

Rank-1 / BatchEnsemble layers that train an efficient ensemble by sharing weights and
learning per-member rank-1 perturbations.

::: geonnax.ensemble

## Heteroscedastic heads

Output heads that model input-dependent observation noise, including Monte-Carlo
sigmoid/softmax approximations.

::: geonnax.heteroscedastic

## Input perturbation

::: geonnax.ncp

## Conditioning & hypernetworks

FiLM modulation, affine conditioners, and hypernetworks that generate the parameters of
a target network (e.g. a conditioned SIREN).

::: geonnax.conditioning
