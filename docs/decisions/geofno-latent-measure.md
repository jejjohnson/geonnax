# Decision record: Geo-FNO latent measure and fold-over regulariser

- **Status:** accepted
- **Issue:** [#85](https://github.com/jejjohnson/geonnax/issues/85), under epic [#76](https://github.com/jejjohnson/geonnax/issues/76)
- **Related:** [#84](https://github.com/jejjohnson/geonnax/issues/84) (`GeoFNO`), [jejjohnson/pyrox#298](https://github.com/jejjohnson/pyrox/issues/298) (a GP prior on the chart)

## Context

`GeoFNO` (#84) maps the points $x_n$ onto the latent torus with a learned chart $\xi = \varphi(x)$ and estimates the Fourier coefficients of the lifted field there:

$$
\hat v_k = \int_{\mathbb{T}^d} v(\xi)\, e^{-2\pi i k\cdot\xi}\, d\xi
\;\approx\; \sum_n w_n\, v_n\, e^{-2\pi i k\cdot\xi_n}.
$$

The physical quadrature weights $w_n$ approximate $dx$, not $d\xi$. By change of variables, $d\xi = \lvert\det\nabla\varphi(x)\rvert\, dx$, so the consistent latent weights are

$$
w_n^{\xi} = \frac{w_n\, \lvert\det\nabla\varphi(x_n)\rvert}{\sum_m w_m\, \lvert\det\nabla\varphi(x_m)\rvert}.
$$

`GeoFNO(jacobian_weights=True)` applies this reweighting, using `LearnedDeformation.jacobian_det`. The reference Geo-FNO [1, 2] uses no weights at all. It also trains its chart without any invertibility constraint, so the chart may fold over ($\det\nabla\varphi \le 0$).

Two questions follow:

1. Should `jacobian_weights` default to `True`?
2. Should a fold-over penalty $\mathbb{E}_x[\mathrm{relu}(-\det\nabla\varphi(x))]$ ship, and if so, where?

## Options considered

- **A. Both off by default** (the state after #84). This is faithful to the published method. The penalty is a few lines in the user's loss.
- **B. `jacobian_weights=True` by default, no built-in penalty.** The quadrature is consistent. However, every forward pass gets a per-point Jacobian (`jacfwd`, $d$ extra JVPs), and the chart's gradient is coupled to the weights.
- **C. A geonnax helper `deformation_fold_penalty(deform, x, code=None)`.** It is cheap and composable, but geonnax has no loss functions. The penalty might belong in pyrox instead, where the prior of pyrox#298 already discourages folding.

## Ablation

**Setup.** The domain is the unit disc.

- **Point density.** Points are drawn with density $p(x) \propto 0.1 + \exp(-\lvert x - (0.45, 0.25)\rvert^2 / (2 \cdot 0.25^2))$. Each sample has its own cloud of $N = 256$ points, about 60 % of them within two blob widths of the centre, a region covering a quarter of the disc.
- **Weights.** The quadrature weights are $w_n \propto 1/p(x_n)$, normalised to sum to 1.
- **Input.** The input $u$ is a random sum of six cosines, $\sum_j a_j \cos(\omega_j\cdot x + \phi_j)$ with $\omega_j \sim \mathcal{N}(0, 2.5^2 I)$.
- **Target.** The target is the nonlinear integral operator $v = \int_{\text{disc}} e^{-\lvert x-y\rvert^2/(2\cdot 0.3^2)} u(y)\,dy + \tfrac12 u^2$. The integral is computed on a uniform reference grid of 2 809 points.
- **Model.** `GeoFNO(1, 1, n_modes=(6, 6), latent_shape=(16, 16), hidden_channels=16, n_layers=3, deformation="mlp", deformation_kwargs={"width": 32, "depth": 2})`.
- **Training.** 256 training and 64 test samples, Adam at $3\times10^{-3}$ with cosine decay, batch 16, 500 steps, seeds 0, 1 and 2. The loss is the quadrature-weighted relative L2.
- **Penalty.** The fold penalty is $\lambda\,\mathrm{mean}(\mathrm{relu}(-\det\nabla\varphi / \det_0))$ with $\lambda = 0.1$. Here $\det_0 = \prod_a \mathrm{scale}_a$ is the determinant at init, so the normalised determinant is 1 for the untrained chart.
- **Reference arm.** `deformation="identity"` is included as a reference.
- **Runtime.** CPU only, on a shared 16-core VM. A full run took about 4 minutes, including data generation and compilation.

The determinant statistics are over all test points of all three seeds, normalised by $\det_0$. "Folds" is the fraction of test points with $\det\nabla\varphi \le 0$, given as the mean over seeds (the maximum in brackets). The test relative L2 is mean ± standard deviation over seeds. The time per step is the median wall time of a jitted training step, excluding the first steps.

| chart | `jacobian_weights` | fold penalty | test rel. L2 | folds | min det J | median det J | ms / step |
|---|---|---|---|---|---|---|---|
| identity | False | off | 0.334 ± 0.006 | 0 % | 1.000 | 1.000 | 20.2 |
| mlp | False | off | **0.327 ± 0.009** | 2.0 % (2.4 %) | −0.95 | 0.77 | 27.4 |
| mlp | False | on | 0.341 ± 0.013 | 14.7 % (19.4 %) | −0.94 | 0.14 | 29.5 |
| mlp | True | off | 0.355 ± 0.006 | 44.8 % (46.4 %) | −12.0 | 0.87 | 33.4 |
| mlp | True | on | 0.333 ± 0.006 | **0 %** | +0.018 | 0.86 | 34.6 |

**Reading.**

- **Jacobian weights alone are harmful.**
    - They raise the test error by about 3 standard deviations (0.355 against 0.327).
    - They make the chart fold over almost half the domain, with determinants from −12 to 17.
    - The weights use $\lvert\det\nabla\varphi\rvert$, so a fold costs nothing. Meanwhile the chart can now steer the quadrature, and it uses that freedom to reweight the points rather than to transport them. This is the instability that option B warned about, and it appears on a single small problem.
- **Jacobian weights need the fold penalty.**
    - With the penalty, the chart has no folds on any seed (min det J > 0).
    - The error returns to the level of the identity chart (0.333).
    - It is still no better than the unweighted chart.
- **Without Jacobian weights, the penalty at this strength does not help.**
    - Mild folds remain (2 % without the penalty; 15 % with it).
    - With the penalty, the chart drifts towards $\det \approx 0$ (median 0.14).
    - The error rises slightly.
    - A relu penalty only acts on points that are already folded, so this is a weak, poorly targeted constraint. A stronger $\lambda$ or a log-barrier would behave differently. That is a tuning question for the user's loss, not a default for the library.
- **Cost.** Jacobian weights add about 20 % per training step at $d = 2$ (27 → 33 ms). Each penalty evaluation adds another `jacobian_det` call.
- **The learned chart is worth little here.** On this problem the best MLP chart (0.327) is within about one standard deviation of the identity chart (0.334). The measure question is therefore second-order compared with the chart itself.

!!! note
    #85 planned the ablation on the Geo-FNO elasticity point-cloud benchmark. This record uses a CPU-cheap synthetic disc problem with deliberately non-uniform density instead, so that the chart has a reason to deform. The conclusion favours the status quo, so a larger benchmark is not needed before this decision. Reopen #85 if a real dataset shows the reweighting helping.

## Decision

- **Keep `jacobian_weights=False` as the default (option A).**
    - Flipping it made the model worse and produced heavily folded charts.
    - It only matched the default once a fold penalty was added.
    - It costs about 20 % more per step.
    - The flag stays available for users who want the consistent measure, but it should be turned on only together with a fold penalty.
- **The fold regulariser lives in the documentation, not in geonnax code (no option C).**
    - geonnax has no losses. The penalty is two lines built on the public `LearnedDeformation.jacobian_det`, and its strength and form (relu or log-barrier) are problem-dependent.
    - The recipe is the snippet below.
    - In pyrox, a prior on the chart (pyrox#298) is the principled way to discourage folds. pyrox needs no new loss helper either.

```python
import jax
import jax.numpy as jnp


def fold_penalty(model, x, code=None):
    """mean(relu(-det∇φ / det₀)) over one cloud; 0 for a fold-free chart."""
    det0 = jnp.prod(model.deformation.scale)  # det∇φ of the untrained box map
    det = model.deformation.jacobian_det(x, code) / det0
    return jnp.mean(jax.nn.relu(-det))


# loss = data_loss + 0.1 * jnp.mean(jax.vmap(fold_penalty, (None, 0))(model, x))
```

## Consequences

- No code change: the `jacobian_weights` default stays `False`, and downstream pyrox and kernellib are unaffected.
- No new public name. `LearnedDeformation.jacobian_det` remains both the fold diagnostic and the building block for the penalty.
- Users who set `jacobian_weights=True` should add the fold penalty above. Without it, the chart exploits the $\lvert\det\rvert$ weights and folds.
- #85 is closed. Reopen it if a real benchmark, such as elasticity or #90, shows the latent measure helping.

## References

1. Z. Li, D. Z. Huang, B. Liu, A. Anandkumar. *Fourier Neural Operator with Learned Deformations for PDEs on General Geometries.* Journal of Machine Learning Research 24(388):1–26, 2023. [arXiv:2207.05209](https://arxiv.org/abs/2207.05209), [JMLR](https://www.jmlr.org/papers/v24/23-0064.html).
2. Reference implementation: [neuraloperator/Geo-FNO](https://github.com/neuraloperator/Geo-FNO) (`elasticity/elas_geofno.py`, the `IPHI` chart).
