"""Plot OVS-F1 against laptop latency for the seven released checkpoints.

OVS-F1 and parameter counts are read from the model comparison table in
docs/results.md. Latency is the median of the raw RTX 3080 Laptop samples
(TensorRT FP32, 243 predicates) that update_readme_latency.py also reads;
the table's laptop column must agree with them. Pass --source with the local
result file; the figure is written to assets/model_family.png.
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from update_readme_latency import MODELS, ROOT, load_records, stats  # noqa: E402

ARM, PREDICATES = "TensorRT FP32", 243
TABLE = ROOT / "docs/results.md"
OUT = ROOT / "assets/model_family.png"
LAPTOP_COLUMN = "RTX 3080 Laptop, TensorRT"
RECOMMENDED = "maelic/relsgg-vits16plus"
COLOR = {"ViT": "#1F6FD0", "ConvNeXt": "#C2571A"}
MARKER = {"ViT": "o", "ConvNeXt": "s"}
INK, GRID, BAND = "#1F2933", "#E3E7EC", "#D5DBE3"
# Label placement in points, chosen for the measured layout.
OFFSET = {
    "ViT-S": (-7, -3),
    "ViT-S+": (7, -16),
    "ViT-B": (7, -3),
    "ConvNeXt-T": (-7, -3),
    "ConvNeXt-S": (7, 2),
    "ConvNeXt-B": (7, -9),
    "ConvNeXt-L": (-7, 5),
}


def read_table():
    """Rows of the model comparison table, keyed by Hugging Face id."""
    lines = TABLE.read_text().split("## Model comparison", 1)[1].splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("|"))
    rows = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        rows.append(line)
    header = [c.strip() for c in rows[0].strip("|").split("|")]
    table = {}
    for line in rows[2:]:
        cells = dict(
            zip(header, (c.strip().strip("*") for c in line.strip("|").split("|")))
        )
        repo = "maelic/" + re.search(r"`(relsgg-[^`]+)`", cells["model"]).group(1)
        table[repo] = cells
    if set(table) != set(MODELS):
        raise ValueError("The model comparison table must list the seven checkpoints")
    return table


def number(cell):
    return float(re.match(r"[\d.]+", cell).group())


def points(records, table):
    out = []
    for record in records:
        repo = record["checkpoint"]
        latency = stats(record, ARM, PREDICATES)[0]
        cells = table[repo]
        if f"{latency:.1f} ms" != cells[LAPTOP_COLUMN]:
            raise SystemExit(
                f"{TABLE.relative_to(ROOT)}: {repo} laptop latency is "
                f"{cells[LAPTOP_COLUMN]!r}, the samples give {latency:.1f} ms"
            )
        label = MODELS[repo]
        out.append(
            dict(
                repo=repo,
                label=label,
                ms=latency,
                ovs=number(cells["OVS-F1"]),
                params=number(cells["params"]),
                family="ViT" if label.startswith("ViT") else "ConvNeXt",
            )
        )
    return out


def frontier(pts):
    """Checkpoints no other checkpoint beats on both latency and OVS-F1."""
    keep = [
        p
        for p in pts
        if not any(
            q["ms"] <= p["ms"]
            and q["ovs"] >= p["ovs"]
            and (q["ms"], q["ovs"]) != (p["ms"], p["ovs"])
            for q in pts
        )
    ]
    return sorted(keep, key=lambda p: p["ms"])


def use_font():
    """Fira Sans when TeX Live ships it, matplotlib's default otherwise."""
    paths = glob.glob(
        "/usr/local/texlive/*/texmf-dist/fonts/opentype/public/fira/FiraSans-*.otf"
    )
    for path in paths:
        fm.fontManager.addfont(path)
    if paths:
        plt.rcParams["font.family"] = "Fira Sans"


def plot(pts, gpu):
    use_font()
    front = frontier(pts)
    fig, ax = plt.subplots(figsize=(7.0, 3.7), dpi=200)
    ax.plot(
        [p["ms"] for p in front],
        [p["ovs"] for p in front],
        color=BAND,
        lw=11,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=1,
    )
    for family in ("ViT", "ConvNeXt"):
        sel = [p for p in pts if p["family"] == family]
        ax.plot(
            [p["ms"] for p in sel],
            [p["ovs"] for p in sel],
            color=COLOR[family],
            lw=1.3,
            alpha=0.5,
            zorder=2,
        )
        ax.scatter(
            [p["ms"] for p in sel],
            [p["ovs"] for p in sel],
            s=70,
            marker=MARKER[family],
            color=COLOR[family],
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )
    for p in pts:
        dx, dy = OFFSET[p["label"]]
        star = ", recommended" if p["repo"] == RECOMMENDED else ""
        ax.annotate(
            f"{p['label']} ({p['params']:.0f} M{star})",
            (p["ms"], p["ovs"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            color=INK,
            ha="left" if dx > 0 else "right",
            fontweight="bold" if star else "normal",
        )
    ax.set_xlabel(
        f"latency per image, ms ({gpu}, TensorRT FP32, batch 1)", fontsize=10, color=INK
    )
    ax.set_ylabel("OVS-F1 (%)", fontsize=10, color=INK)
    ax.tick_params(labelsize=9, colors=INK)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    xs, ys = [p["ms"] for p in pts], [p["ovs"] for p in pts]
    ax.set_xlim(min(xs) - 7.5, max(xs) + 2.5)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    handles = [Line2D([], [], color=BAND, lw=6, label="Pareto frontier")] + [
        Line2D(
            [],
            [],
            color=COLOR[f],
            marker=MARKER[f],
            ms=7,
            lw=1.3,
            markeredgecolor="white",
            label=f"DINOv3 {f}",
        )
        for f in ("ViT", "ConvNeXt")
    ]
    ax.legend(handles=handles, fontsize=9, frameon=False, loc="lower right")
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    print(
        f"Wrote {OUT.relative_to(ROOT)}; frontier: {', '.join(p['label'] for p in front)}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="local benchmark records JSON"
    )
    args = parser.parse_args()
    records = load_records(args.source)
    gpu = records[0]["gpu"].removeprefix("NVIDIA GeForce ")
    plot(points(records, read_table()), gpu)


if __name__ == "__main__":
    main()
