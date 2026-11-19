"""Render the example's figures from the docs' own code.

The figures run the Python blocks of docs/guides/methane.md, verbatim and in
order, then plot what they computed, so the numbers quoted in the page and the
pictures come from the same run. The basis gallery moved to functax's README
(jejjohnson/geonnax#160).

Run from the repo root (about 15 minutes on a laptop CPU; optax is needed by
the example only):

    uv run --with optax python docs/assets/readme/figures.py
"""

from __future__ import annotations

import re
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import geonnax as gnx


OUT = Path(__file__).parent
METHANE = OUT.parents[1] / "guides" / "methane.md"


def example_blocks() -> list[str]:
    """The ```python blocks of the methane example page, in order."""
    return re.findall(r"```python\n(.*?)```", METHANE.read_text(), flags=re.S)


# ---------------------------------------------------------------- plotting

NLAT, NLON = 90, 180
LON, LAT = np.meshgrid(np.linspace(-179, 179, NLON), np.linspace(-89, 89, NLAT))


def moll(ax, field, *, title, cmap="viridis", vmin=None, vmax=None, shape=None):
    lat, lon = (LAT, LON) if shape is None else shape
    ax.pcolormesh(
        np.deg2rad(lon),
        np.deg2rad(lat),
        np.asarray(field).reshape(lat.shape),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        rasterized=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, lw=0.3, color="k", alpha=0.25)
    ax.set_title(title, fontsize=9)


def row(n, width=2.75, height=1.75):
    return plt.subplots(
        1, n, figsize=(width * n, height), subplot_kw={"projection": "mollweide"}
    )


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- example


def example_figures() -> dict:
    ns: dict = {}
    for block in example_blocks():
        exec(compile(block, METHANE.name, "exec"), ns)

    x, lonlat, mask = ns["x"], ns["lonlat"], ns["mask"]
    lo, hi = float(x.min()), float(x.max())

    # Stage 1: what each encoder can express.
    fig, axes = row(5)
    moll(axes[0], x, title="truth x (ppb)", vmin=lo, vmax=hi)
    for ax, l_max in zip(axes[1:4], (8, 16, 28), strict=True):
        Z = jax.vmap(gnx.SphericalHarmonicEncoder(l_max, input_mode="lonlat"))(lonlat)
        fit = Z @ jnp.linalg.lstsq(Z, x)[0]
        rmse = ns["rmse_sh"][l_max]
        moll(ax, fit, title=f"harmonics l ≤ {l_max}: {rmse:.1f} ppb", vmin=lo, vmax=hi)
    in_R = np.asarray(ns["in_R"])
    cap = np.full(x.shape, np.nan)
    cap[in_R] = np.asarray(ns["G"][in_R] @ ns["gamma"] + jnp.mean(x[ns["in_R"]]))
    moll(
        axes[4],
        cap,
        title=f"Slepian, 20° cap: {ns['rmse_cap']:.2f} ppb",
        vmin=lo,
        vmax=hi,
    )
    save(fig, "encode.png")

    # Stage 2: three coordinate networks, scored off the training grid.
    shape1 = (np.asarray(ns["lat1"]), np.asarray(ns["lon1"]))
    fig, axes = row(4)
    moll(axes[0], ns["x1"], title="truth on a 1° grid", vmin=lo, vmax=hi, shape=shape1)
    for ax, (name, f) in zip(axes[1:], ns["fits"].items(), strict=True):
        pred = 1880.0 + 25.0 * jax.vmap(f)(ns["lonlat1"])
        title = f"{name}: {ns['rmse_net'][name]:.1f} ppb"
        moll(ax, pred, title=title, vmin=lo, vmax=hi, shape=shape1)
    save(fig, "represent.png")

    # Stage 3: the GP head's mean and spread, from the tracks only.
    obs = np.full(x.shape, np.nan)
    obs[np.asarray(mask)] = np.asarray(ns["y"])
    fig, axes = row(3, width=3.4, height=2.1)
    moll(axes[0], obs, title="one day of tracks, y (ppb)", vmin=lo, vmax=hi)
    rmse = ns["rmse_gp"]
    title = f"mean x̂: {rmse['observed']:.1f} / {rmse['unobserved']:.1f} ppb"
    moll(axes[1], ns["x_hat"], title=title, vmin=lo, vmax=hi)
    moll(axes[2], ns["sigma"], title="spread σ (relative)", cmap="magma")
    save(fig, "head.png")
    return ns


def report(ns: dict) -> None:
    """Print the numbers the methane example quotes."""
    for key in ("rmse_sh", "rmse_cap", "rmse_net", "rmse_gp", "sigma_ratio"):
        print(f"{key}: {ns[key]}")
    print(f"observed cells O = {int(ns['mask'].sum())}")


if __name__ == "__main__":
    report(example_figures())
