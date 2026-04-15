#!/usr/bin/env python3
"""
PluMA-contract verification for SHAPExplainability.

Runs the plugin end-to-end on example/ and compares each generated
output file against its example/<name>.expected twin using the same
numeric-tolerant comparison logic as PluMA/testPluMA.py (EPS=1e-8).

Exits 0 if all .expected files match, nonzero otherwise, with a diff
printed per mismatch.

Run from the repo root:

    python scripts/verify_pluma.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = REPO_ROOT / "example"

sys.path.insert(0, str(REPO_ROOT))

EPS = 1e-8


def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def check_accuracy(generated: Path, expected: Path) -> bool:
    """Numeric-tolerant line-by-line comparison, ported from testPluMA.py."""
    lines1 = generated.read_text().splitlines()
    lines2 = expected.read_text().splitlines()
    lines1.sort()
    lines2.sort()

    if len(lines1) != len(lines2):
        print(f"  line count differs: generated={len(lines1)} expected={len(lines2)}")
        return False

    filetype = "CSV" if generated.suffix == ".csv" else "TXT"

    for i, (a, b) in enumerate(zip(lines1, lines2)):
        if filetype == "CSV":
            d1 = a.split(",")
            d2 = b.split(",")
        else:
            d1 = a.split()
            d2 = b.split()
        if len(d1) != len(d2):
            print(f"  line {i}: field count differs {len(d1)} vs {len(d2)}")
            print(f"    generated: {a!r}")
            print(f"    expected : {b!r}")
            return False
        for j, (x, y) in enumerate(zip(d1, d2)):
            if not is_number(x):
                if x != y:
                    print(f"  line {i} field {j}: {x!r} != {y!r}")
                    return False
            else:
                if abs(float(x) - float(y)) > EPS:
                    print(f"  line {i} field {j}: |{x} - {y}| > {EPS}")
                    return False
    return True


def run_plugin() -> None:
    """Invoke the plugin the same way PluMA would."""
    from SHAPExplainabilityPlugin import SHAPExplainabilityPlugin

    params = EXAMPLE / "parameters.verify.txt"
    params.write_text(
        f"model\t{EXAMPLE}/model.joblib\n"
        f"features\t{EXAMPLE}/features.csv\n"
        f"labels\t{EXAMPLE}/labels.csv\n"
        "explainer\ttree\n"
        "background_samples\t50\n"
        "n_top_features\t10\n"
        "compute_interactions\tfalse\n"
    )

    plugin = SHAPExplainabilityPlugin()
    plugin.input(str(params))
    plugin.run()
    plugin.output(str(EXAMPLE / "output"))
    params.unlink()


def main() -> int:
    # Clean any stale outputs (but keep .expected files)
    for p in EXAMPLE.glob("output.*"):
        if not p.name.endswith(".expected"):
            p.unlink()

    print("Running SHAPExplainabilityPlugin against example/ fixture...")
    run_plugin()
    print("Done. Comparing against .expected files...\n")

    expected_files = sorted(EXAMPLE.glob("output.*.expected"))
    if not expected_files:
        print("No .expected files found under example/", file=sys.stderr)
        return 2

    any_fail = False
    for expected in expected_files:
        # "output.shap_values.csv.expected" -> "output.shap_values.csv"
        generated = expected.with_suffix("")
        if not generated.exists():
            print(f"[FAIL] {generated.name} did not generate")
            any_fail = True
            continue
        ok = check_accuracy(generated, expected)
        print(f"[{'PASS' if ok else 'FAIL'}] {generated.name}")
        if not ok:
            any_fail = True

    print()
    if any_fail:
        print("Verification FAILED")
        return 1
    print("All outputs match expected - PluMA contract verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
