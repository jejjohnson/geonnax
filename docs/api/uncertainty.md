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

## Lipschitz-bounded layers

Spectral normalisation rescales a linear layer's weight to a fixed spectral norm,
$\hat W = c\,W / \hat\sigma(W)$, making the layer approximately $c$-Lipschitz —
$\hat\sigma$ is a power-iteration estimate that approaches $\sigma(W)$ from below, so
the bound is the cheap, standard one rather than a provable one. It is the wrapper half of
the SNGP / DUE recipe: `RandomFeatureGaussianProcess` above supplies the distance-aware
output head, and spectrally normalising the upstream dense layers is what makes the
feature extractor distance-preserving enough for that head to be meaningful.

::: geonnax.spectral_norm

## Deep ensembles

Rank-1 / BatchEnsemble layers that train an efficient ensemble by sharing weights and
learning per-member rank-1 perturbations.

::: geonnax.ensemble

## Heteroscedastic heads

Output heads that model input-dependent observation noise, including Monte-Carlo
sigmoid/softmax approximations.

::: geonnax.heteroscedastic

## Mixture-density & structured-prediction heads

Edward2-style output heads. The mixture-density head maps a feature vector to
$(\pi, \mu, \log\sigma)$ for a diagonal Gaussian mixture — multi-modal regression on
top of any backbone. The linear-chain CRF supplies the forward algorithm and Viterbi
decoding for sequence tagging. Both return parameters or decoded values rather than
distribution objects, so downstream libraries can attach priors.

::: geonnax.mixture

::: geonnax.crf

## Input perturbation

::: geonnax.ncp

## Conditioning & hypernetworks

FiLM modulation, affine conditioners, and hypernetworks that generate the parameters of
a target network (e.g. a conditioned SIREN).

::: geonnax.conditioning
