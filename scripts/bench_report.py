#!/usr/bin/env python3
"""Turn bench_hybrid.sh's CSV into a Markdown report.

Reports best-of-N rather than the mean: the fastest run is the one least
disturbed by whatever else touched the machine. The spread across repetitions
is printed too, so a reader can judge whether the gaps are real.
"""
import csv, sys
from collections import defaultdict


def main(path: str) -> None:
    runs = defaultdict(list)
    grids = set()
    res = defaultdict(set)          # residual per rank count, not overall
    with open(path) as fh:
        for row in csv.DictReader(fh):
            key = (row["backend"], int(row["ranks"]))
            runs[key].append((float(row["setup"]), float(row["solve"]), float(row["total"])))
            grids.add(row["grid"])
            res[int(row["ranks"])].add(row["relres"])

    if not runs:
        sys.exit(f"{path}: no data rows")

    ranks = sorted({r for _, r in runs})
    backends = [b for b in ("openmpi", "tmpi") if any(k[0] == b for k in runs)]
    label = {"openmpi": "OpenMPI (processes)", "tmpi": "thread-MPI (threads)"}

    best = {k: min(v, key=lambda t: t[2]) for k, v in runs.items()}
    spread = max(
        (max(t[2] for t in v) - min(t[2] for t in v)) / min(t[2] for t in v)
        for v in runs.values() if len(v) > 1
    ) if any(len(v) > 1 for v in runs.values()) else 0.0

    print(f"Grid {'/'.join(sorted(grids))}^3, best of {max(len(v) for v in runs.values())} runs; "
          f"worst run-to-run spread {spread * 100:.1f}%.")
    # The residual legitimately changes with the rank count -- the partitioning
    # changes, so AMG does not build the same hierarchy. What must not change is
    # the residual between the two backends at the SAME rank count.
    disagree = {n: sorted(v) for n, v in res.items() if len(v) > 1}
    if disagree:
        print(f"\n**Backends disagree at these rank counts: {disagree}**\n")
    else:
        print("Both backends reach an identical relative residual at every rank "
              "count (" + ", ".join(f"{n}: {sorted(v)[0]}" for n, v in sorted(res.items())) + ").\n")

    print("| ranks | " + " | ".join(f"{label[b]} total (s)" for b in backends) +
          (" | thread-MPI vs OpenMPI |" if len(backends) == 2 else " |"))
    print("|---" * (len(backends) + 1 + (len(backends) == 2)) + "|")
    for n in ranks:
        cells = []
        for b in backends:
            t = best.get((b, n))
            cells.append(f"{t[2]:.2f}" if t else "—")
        row = f"| {n} | " + " | ".join(cells)
        if len(backends) == 2:
            a, c = best.get(("openmpi", n)), best.get(("tmpi", n))
            row += f" | {a[2] / c[2]:.2f}x |" if a and c else " | — |"
        else:
            row += " |"
        print(row)

    print("\n<details><summary>setup vs solve</summary>\n")
    print("| backend | ranks | setup (s) | solve (s) |")
    print("|---|---|---|---|")
    for b in backends:
        for n in ranks:
            t = best.get((b, n))
            if t:
                print(f"| {label[b]} | {n} | {t[0]:.2f} | {t[1]:.2f} |")
    print("\n</details>")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bench_hybrid.csv")
