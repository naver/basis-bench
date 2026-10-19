'''
BASIS-Bench
    Copyright (C) 2026 NAVER Corporation.
    Apache-2.0
'''

import glob
import json
import os


def load_inputs(paths):
    """Load one or more result files.

    Each path may be a file, a directory (searched recursively for *.json) or
    a glob. Returns a list of (path, items).
    """
    files = []
    for path in paths:
        if os.path.isdir(path):
            files.extend(sorted(glob.glob(os.path.join(path, "**", "*.json"),
                                          recursive=True)))
        elif os.path.isfile(path):
            files.append(path)
        else:
            matched = sorted(glob.glob(path))
            if not matched:
                raise FileNotFoundError(f"no such file, directory or match: {path}")
            files.extend(matched)

    out = []
    for path in files:
        with open(path) as f:
            out.append((path, json.load(f)))
    return out


def print_table(rows, columns, title=None):
    """Print a fixed-width table. Floats are shown with one decimal."""
    if not rows:
        print("(no rows)")
        return

    def cell(value):
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.1f}"
        return str(value)

    widths = {
        c: max(len(c), max(len(cell(r.get(c))) for r in rows)) for c in columns
    }

    if title:
        print()
        print(title)
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(cell(row.get(c)).ljust(widths[c]) for c in columns))


def save_json(obj, path):
    if not path:
        return
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)
    print(f"\nSaved -> {path}")
