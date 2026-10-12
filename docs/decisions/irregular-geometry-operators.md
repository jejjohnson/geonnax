# Decision record: OTNO, GINO and Transolver

- **Status:** accepted
- **Issue:** [#91](https://github.com/jejjohnson/geonnax/issues/91), under epic [#78](https://github.com/jejjohnson/geonnax/issues/78) (wave [#74](https://github.com/jejjohnson/geonnax/issues/74))
- **Related:** #89 (FNO on curvilinear grids), #90 (Western Mediterranean SSH three ways), the deferred graph neural operator (GNO)

## Context

Wave #74 adds neural operators for fields on irregular geometry. It offers three routes, chosen by what varies:

- Structured curvilinear grid with fixed geometry: plain `FNO` in index space, coordinates as input channels (docs only, #89).
- Fixed unstructured geometry: a basis fitted to the domain (theme C).
- Geometry that varies per sample: deform onto a torus and use an FFT in the latent grid (Geo-FNO, themes A and B).

Three further operators from the literature target irregular geometry. This record fixes which of them geonnax implements, so the question is not reopened each quarter. In each case the question is also where the geometry processing belongs: model code or data preparation.

## Options considered

### OTNO (optimal transport neural operator)

Li, Li, Kovachki and Anandkumar [1] compute a per-shape optimal transport plan (Sinkhorn) or map (PPMM) that carries the surface mesh onto a latent torus grid. The solve is done offline, once per shape. The model itself is FNO blocks followed by a gather with the precomputed index map (`ind_dec` in neuraloperator). The reported gain on ShapeNet-Car pressure (relative L2 6.70 % against 7.21 % for GINO and 13.50 % for Geo-FNO, with 1.9 GB against 12.7 GB of memory) comes from the per-shape chart, which is preprocessing.

- **A. Implement OTNO in geonnax.** Nothing would be added beyond `FNO` and a gather, and the OT solve needs POT or geomloss, which are not geonnax dependencies.
- **B. No geonnax code.** Document the chart construction as a recipe and put the OT solve on the data side (geotoolz).

### GINO (geometry-informed neural operator)

Li et al. [2] use a GNO encoder (radius-graph kernel integral) onto a regular latent grid, a latent FNO, a GNO decoder, and signed-distance features.

- **A. Implement GINO now.** This forces the GNO decision early, for 3-D CFD-scale problems that are not this stack's main use case.
- **B. Defer with GNO.** GNO was deferred because of neighbour search and ragged neighbourhoods, which fit JAX poorly. GINO depends on it.

### Transolver

Wu et al. [3] softly assign N points to S learned slices, run attention among the S slice tokens, and scatter back. The cost is O(N S), shapes are fixed, and no neighbour search is needed, so it suits JAX. It would build on `geonnax.layers.Attention`.

- **A. Implement now.** It is a learned-basis method, and nothing yet shows it beats `GeoFNO` on the data this stack targets.
- **B. Evaluate after #90.** Open a `research:` issue once #90 reports, with #90's benchmark as the bar to beat.

## Decision

- **OTNO: no geonnax code.** Write a "chart from optimal transport" recipe in the #89 guide, and open a geotoolz issue for the OT preprocessing.
- **GINO: deferred together with GNO.** Revisit only when GNO is revisited.
- **Transolver: research issue after #90.** Implement only if it beats the #90 benchmark.

## Consequences

- No new runtime dependency.
- The OT chart is computed outside geonnax; the model side stays `FNO` plus a gather.
- GINO has no home until GNO is reopened.
- The question is closed until #90 reports.

!!! note
    The repository has no earlier decision-record location and no written GNO deferral record. This page starts `docs/decisions/`. The GNO deferral is described here only as far as issue #91 states it.

## References

1. X. Li, Z. Li, N. Kovachki, A. Anandkumar. *Geometric Operator Learning with Optimal Transport.* NeurIPS 2025; JMLR 26(282):1-38, 2025. [arXiv:2507.20065](https://arxiv.org/abs/2507.20065), [JMLR](https://www.jmlr.org/papers/v26/25-1380.html).
2. Z. Li, N. Kovachki, C. Choy, B. Li, J. Kossaifi, S. Otta, M. A. Nabian, M. Stadler, C. Hundt, K. Azizzadenesheli, A. Anandkumar. *Geometry-Informed Neural Operator for Large-Scale 3D PDEs.* NeurIPS 2023. [arXiv:2309.00583](https://arxiv.org/abs/2309.00583).
3. H. Wu, H. Luo, H. Wang, J. Wang, M. Long. *Transolver: A Fast Transformer Solver for PDEs on General Geometries.* ICML 2024. [arXiv:2402.02366](https://arxiv.org/abs/2402.02366).
