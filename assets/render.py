"""Render the geonnax README assets: icon, logos and the two diagrams.

Every output is a plain SVG written next to this file. The icon shares the
gaussx, kernellib and pyrox tile (a 120-unit rounded square, r = 26, white
glyph) in the geonnax green gradient. Its glyph is a globe whose equator is a
sinusoid: a signal on the sphere.

Diagrams come in a light and a dark variant for GitHub's
``<picture>`` / ``prefers-color-scheme`` switch.

Run from the repo root:

    uv run --no-project python docs/assets/render.py
"""

import math
from pathlib import Path


OUT = Path(__file__).parent

DOT = " \N{MIDDLE DOT} "
SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

# Green: the geonnax gradient and its wordmark accents.
C1, C2 = "#22C55E", "#15803D"
ACCENT = {"light": "#15803D", "dark": "#4ADE80"}

THEME = {
    "light": {
        "text": "#1F2937",
        "muted": "#6B7280",
        "card": "#F9FAFB",
        "line": "#D1D5DB",
        "arrow": "#9CA3AF",
        "hi_fill": "#F0FDF4",
        "hi_line": "#16A34A",
        "site": "#15803D",
        "band": "#F3F4F6",
    },
    "dark": {
        "text": "#E5E7EB",
        "muted": "#9CA3AF",
        "card": "#111827",
        "line": "#374151",
        "arrow": "#6B7280",
        "hi_fill": "#052E16",
        "hi_line": "#22C55E",
        "site": "#4ADE80",
        "band": "#1F2937",
    },
}

W = "#FFFFFF"


# ---------------------------------------------------------------- glyph


def _wave(x0: float, x1: float, yc: float, amp: float, cycles: int, n: int = 72) -> str:
    pts = []
    for i in range(n + 1):
        x = x0 + (x1 - x0) * i / n
        y = yc - amp * math.sin(2 * math.pi * cycles * i / n)
        pts.append(f"{x:.1f} {y:.1f}")
    return "M" + " L".join(pts)


def glyph_wave_globe() -> str:
    """A globe outline, one meridian, and a sinusoid for its equator."""
    return (
        f'<circle cx="64" cy="64" r="42" fill="none" stroke="{W}" stroke-width="4.5"/>'
        f'<ellipse cx="64" cy="64" rx="17" ry="42" fill="none" stroke="{W}" '
        'stroke-width="2.7" stroke-opacity="0.6"/>'
        f'<path d="{_wave(22, 106, 64, 12, 3)}" fill="none" stroke="{W}" '
        'stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>'
    )


def tile(gid: str, x: float = 0, y: float = 0, size: float = 128) -> str:
    """The gradient tile with the glyph, placed at (x, y), ``size`` wide."""
    s = size / 128
    return (
        f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{C1}"/><stop offset="1" stop-color="{C2}"/>'
        "</linearGradient></defs>"
        f'<g transform="translate({x} {y}) scale({s:g})">'
        f'<rect x="4" y="4" width="120" height="120" rx="26" fill="url(#{gid})"/>'
        f"{glyph_wave_globe()}</g>"
    )


def svg(width: int, height: int, label: str, title: str, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{label}">\n'
        f"<title>{title}</title>\n{body}\n</svg>\n"
    )


def text(
    x: float,
    y: float,
    s: str,
    *,
    size: float = 14,
    weight: int = 400,
    fill: str,
    family: str = SANS,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{family}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'xml:space="preserve">{s}</text>'
    )


def arrow_defs(mid: str, colour: str) -> str:
    return (
        f'<defs><marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="{colour}"/></marker></defs>'
    )


def line(points: list[tuple[float, float]], colour: str, head: str | None) -> str:
    d = " ".join(f"{'M' if i == 0 else 'L'}{x} {y}" for i, (x, y) in enumerate(points))
    end = f' marker-end="url(#{head})"' if head else ""
    return (
        f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="1.6" '
        f'stroke-linejoin="round"{end}/>'
    )


def card(
    x: float,
    y: float,
    w: float,
    h: float,
    t: dict[str, str],
    *,
    highlight: bool = False,
) -> str:
    fill, stroke, sw = (
        (t["hi_fill"], t["hi_line"], 2) if highlight else (t["card"], t["line"], 1)
    )
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"/>'
    )


# ---------------------------------------------------------------- icon and logo


def write_icon() -> None:
    (OUT / "icon.svg").write_text(svg(128, 128, "geonnax", "geonnax", tile("ic")))


def write_logos() -> None:
    for mode, t in THEME.items():
        body = tile(f"logo-{mode}", x=6, y=6) + (
            f'<text x="154" y="92" font-family="{SANS}" font-size="72" '
            f'font-weight="700" letter-spacing="-1.5" fill="{t["text"]}">'
            f'geonna<tspan fill="{ACCENT[mode]}">x</tspan></text>'
        )
        (OUT / f"logo-{mode}.svg").write_text(svg(470, 140, "geonnax", "geonnax", body))


# ---------------------------------------------------------------- hero

# (title, verb, names) for the coordinate-network lane.
STAGES = [
    (
        "encode",
        "coordinates → features",
        [
            "Cartesian3DEncoder",
            "CyclicEncoder",
            "SphericalHarmonicEncoder",
            "SlepianEncoder",
        ],
    ),
    (
        "represent",
        "features → hidden",
        ["SIREN", "MultiScaleSIREN", "FourierNet" + DOT + "GaborNet", "ConditionedINR"],
    ),
    (
        "head",
        "hidden → value, spread",
        [
            "RandomFeatureGaussianProcess",
            "HeteroscedasticHead",
            "MixtureOfGaussiansDenseHead",
            "DeepVSSGPCore",
        ],
    ),
]
OPERATORS = ["UNet", "FNO", "SFNO", "WNO", "MSWT"]
BASES = (
    "spherical harmonics"
    + DOT
    + "Slepian"
    + DOT
    + "needlets"
    + DOT
    + "wavelets"
    + DOT
    + "Fourier"
    + DOT
    + "EOFs"
    + DOT
    + "graph Laplacian"
)


def write_hero() -> None:
    """Two lanes, coordinates → field and field → field, over one basis library."""
    width, height = 960, 492
    for mode, t in THEME.items():
        head = f"hero-{mode}-head"
        a = t["arrow"]
        parts = [arrow_defs(head, a)]

        # Lane 1: coordinate networks.
        parts.append(
            text(20, 26, "Coordinate networks", size=15, weight=700, fill=t["text"])
        )
        parts.append(
            text(186, 26, "x ↦ f(x), one point at a time", size=13, fill=t["muted"])
        )
        y0, h = 44, 172
        parts.append(card(20, y0, 104, h, t))
        parts.append(
            text(
                72,
                y0 + 80,
                "(λ, φ, t)",
                size=14,
                weight=700,
                fill=t["text"],
                family=MONO,
                anchor="middle",
            )
        )
        parts.append(
            text(72, y0 + 104, "one point", size=12, fill=t["muted"], anchor="middle")
        )
        xs = [(146, 226), (394, 200), (616, 236)]
        for (title, verb, names), (x, w) in zip(STAGES, xs, strict=True):
            parts.append(card(x, y0, w, h, t, highlight=True))
            parts.append(
                text(x + 16, y0 + 26, title, size=14, weight=700, fill=t["site"])
            )
            parts.append(text(x + 16, y0 + 46, verb, size=12, fill=t["muted"]))
            for i, n in enumerate(names):
                parts.append(
                    text(
                        x + 16,
                        y0 + 80 + 22 * i,
                        n,
                        size=12.5,
                        weight=600,
                        fill=t["text"],
                        family=MONO,
                    )
                )
        parts.append(card(874, y0, 66, h, t))
        parts.append(
            text(
                907,
                y0 + 80,
                "f(x)",
                size=14,
                weight=700,
                fill=t["text"],
                family=MONO,
                anchor="middle",
            )
        )
        parts.append(
            text(
                907,
                y0 + 104,
                "σ(x)",
                size=13,
                fill=t["muted"],
                family=MONO,
                anchor="middle",
            )
        )
        cy = y0 + h / 2
        for x1, x2 in [(124, 146), (372, 394), (594, 616), (852, 874)]:
            parts.append(line([(x1, cy), (x2 - 2, cy)], a, head))

        # Lane 2: neural operators.
        y1 = 262
        parts.append(
            text(20, y1 - 10, "Neural operators", size=15, weight=700, fill=t["text"])
        )
        parts.append(
            text(
                156,
                y1 - 10,
                "u ↦ 𝒢(u), a whole field at a time",
                size=13,
                fill=t["muted"],
            )
        )
        h2 = 96
        parts.append(card(20, y1 + 8, 104, h2, t))
        parts.append(
            text(
                72,
                y1 + 50,
                "u",
                size=14,
                weight=700,
                fill=t["text"],
                family=MONO,
                anchor="middle",
            )
        )
        parts.append(
            text(
                72,
                y1 + 74,
                "(C, H, W)",
                size=12,
                fill=t["muted"],
                family=MONO,
                anchor="middle",
            )
        )
        parts.append(card(146, y1 + 8, 706, h2, t, highlight=True))
        parts.append(text(162, y1 + 34, "operate", size=14, weight=700, fill=t["site"]))
        parts.append(
            text(
                232,
                y1 + 34,
                "grid → grid, on the plane or the sphere",
                size=12,
                fill=t["muted"],
            )
        )
        step = 690 / len(OPERATORS)
        for i, n in enumerate(OPERATORS):
            parts.append(
                text(
                    162 + step * i + step / 2 - 8,
                    y1 + 74,
                    n,
                    size=15,
                    weight=700,
                    fill=t["text"],
                    family=MONO,
                    anchor="middle",
                )
            )
        parts.append(card(874, y1 + 8, 66, h2, t))
        parts.append(
            text(
                907,
                y1 + 62,
                "𝒢(u)",
                size=14,
                weight=700,
                fill=t["text"],
                family=MONO,
                anchor="middle",
            )
        )
        cy2 = y1 + 8 + h2 / 2
        for x1, x2 in [(124, 146), (852, 874)]:
            parts.append(line([(x1, cy2), (x2 - 2, cy2)], a, head))

        # Shared basis library.
        yb = 406
        parts.append(
            f'<rect x="20" y="{yb}" width="920" height="66" rx="12" '
            f'fill="{t["band"]}"/>'
        )
        parts.append(
            text(
                40,
                yb + 28,
                "geonnax.basis",
                size=14,
                weight=700,
                fill=t["site"],
                family=MONO,
            )
        )
        parts.append(
            text(
                40,
                yb + 50,
                "shared by both lanes, and by kernellib and pyrox-gp",
                size=12,
                fill=t["muted"],
            )
        )
        parts.append(text(920, yb + 40, BASES, size=13, fill=t["text"], anchor="end"))

        label = (
            "geonnax has two families: coordinate networks map a point to a value "
            "through an encoder, a representation network and an optional "
            "uncertainty head; neural operators map a gridded field to a field. "
            "Both draw on one basis library."
        )
        (OUT / f"hero-{mode}.svg").write_text(
            svg(width, height, label, "Coordinates in, fields out", "\n".join(parts))
        )


# ---------------------------------------------------------------- layers

USERS = [
    ("kernellib", "basis functions" + DOT + "random features"),
    ("pyrox-gp", "eigenfunction bases for inducing features"),
    ("pyrox-nn", "Bayesian SIREN, MFN, SNGP, VSSGP"),
]


def write_layers() -> None:
    """Where geonnax sits; arrows point at what each package imports."""
    width, height = 960, 384
    for mode, t in THEME.items():
        head = f"layers-{mode}-head"
        a = t["arrow"]
        parts = [arrow_defs(head, a)]
        xs = [30, 350, 670]
        for (name, what), x in zip(USERS, xs, strict=True):
            parts.append(card(x, 20, 260, 76, t))
            parts.append(
                text(
                    x + 130,
                    52,
                    name,
                    size=16,
                    weight=700,
                    fill=t["text"],
                    family=MONO,
                    anchor="middle",
                )
            )
            parts.append(
                text(x + 130, 76, what, size=12.5, fill=t["muted"], anchor="middle")
            )
            parts.append(
                line(
                    [
                        (x + 130, 96),
                        (
                            x + 130 if x == 350 else (x + 130 + (480 - x - 130) * 0.45),
                            166,
                        ),
                    ],
                    a,
                    head,
                )
            )
        parts.append(card(170, 170, 620, 96, t, highlight=True))
        parts.append(tile(f"layers-{mode}-icon", 192, 188, 60))
        parts.append(
            f'<text x="268" y="222" font-family="{SANS}" font-size="26" '
            'font-weight="700" '
            f'fill="{t["text"]}">geonna<tspan fill="{ACCENT[mode]}">x</tspan></text>'
        )
        parts.append(
            text(
                268,
                248,
                "networks, operators and bases as plain equinox.Modules",
                size=13,
                fill=t["muted"],
            )
        )
        parts.append(line([(480, 266), (480, 312)], a, head))
        parts.append(
            f'<rect x="20" y="316" width="920" height="48" rx="10" fill="{t["band"]}"/>'
        )
        parts.append(
            text(
                480,
                346,
                "jax"
                + DOT
                + "equinox"
                + DOT
                + "jaxtyping"
                + DOT
                + "einx"
                + DOT
                + "einops",
                size=14,
                weight=600,
                fill=t["text"],
                family=MONO,
                anchor="middle",
            )
        )
        label = (
            "kernellib, pyrox-gp and pyrox-nn import geonnax; geonnax depends only on "
            "jax, equinox, jaxtyping, einx and einops."
        )
        (OUT / f"layers-{mode}.svg").write_text(
            svg(width, height, label, "Where geonnax sits", "\n".join(parts))
        )


if __name__ == "__main__":
    write_icon()
    write_logos()
    write_hero()
    write_layers()
