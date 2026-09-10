#!/usr/bin/env python3
"""Convert a jupytext-style percent-format .py file into a Colab-ready .ipynb.

The .py file is the source of truth: it diffs cleanly in git, unlike notebook JSON.
Regenerate the notebook after editing it:

    python tools/py_to_ipynb.py notebooks/skin_cancer_cnn_pytorch.py

Cell markers:
    # %%              -> code cell
    # %% [markdown]   -> markdown cell (leading "# " stripped from each line)
"""
import json
import sys
from pathlib import Path

MARKER = "# %%"


def split_cells(text: str):
    cells, kind, buf = [], None, []
    for line in text.splitlines():
        if line.startswith(MARKER):
            if kind is not None:
                cells.append((kind, buf))
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        elif kind is not None:
            buf.append(line)
    if kind is not None:
        cells.append((kind, buf))
    return cells


def clean(lines):
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def to_source(lines):
    """Notebook JSON stores source as a list of lines, each keeping its trailing newline
    except the last."""
    if not lines:
        return []
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def convert(src: Path, dst: Path) -> int:
    cells = []
    for kind, raw in split_cells(src.read_text()):
        lines = clean(list(raw))
        if not lines:
            continue
        if kind == "markdown":
            lines = [ln[2:] if ln.startswith("# ") else ("" if ln.strip() == "#" else ln)
                     for ln in lines]
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": to_source(lines)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": to_source(lines)})

    nb = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True, "gpuType": "T4"},
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    return len(cells)


if __name__ == "__main__":
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".ipynb")
    n = convert(src, dst)
    print(f"{src} -> {dst}  ({n} cells)")
