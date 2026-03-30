"""
extract.py  —  Compare optimal (.sol) vs Gillett Sweep vs Enhanced
Writes comparison_results.txt in the current directory.
"""

import os, glob


# ── Cost extraction ───────────────────────────────────────────────────

def extract_cost(line: str) -> float:
    """Return the last numeric token in a line, or -1 if none found."""
    val = -1.0
    for tok in line.split():
        try:
            v = float(tok)
            if v > 0:
                val = v
        except ValueError:
            pass
    return val


def read_cost(filepath: str) -> float:
    """Open a file and return the first cost value found, or -1."""
    try:
        with open(filepath) as f:
            for line in f:
                if "Cost" in line or "cost" in line:
                    v = extract_cost(line)
                    if v > 0:
                        return v
    except FileNotFoundError:
        pass
    return -1.0


# ── Table helpers ─────────────────────────────────────────────────────

W0, W1, W2, W3, W4, W5 = 22, 12, 14, 14, 10, 10

def sep() -> str:
    return (f"+{'-'*(W0+2)}+{'-'*(W1+2)}+{'-'*(W2+2)}"
            f"+{'-'*(W3+2)}+{'-'*(W4+2)}+{'-'*(W5+2)}+\n")

def cell(s: str, w: int) -> str:
    return f" {s:<{w}} "

def cell_d(v: float, w: int, prec: int = 2) -> str:
    s = "N/A" if v < 0 else f"{v:.{prec}f}"
    return f" {s:<{w}} "


# ── Main ──────────────────────────────────────────────────────────────

def main():
    vrp_files = sorted(glob.glob("*.vrp"))
    if not vrp_files:
        print("No .vrp files found in current directory.")
        return

    instances = [f[:-4] for f in vrp_files]   # strip .vrp

    out_path = "comparison_results.txt"
    with open(out_path, "w") as out:

        # Header
        out.write(sep())
        out.write(f"|{cell('Instance', W0)}"
                  f"|{cell('Optimal (.sol)', W1)}"
                  f"|{cell('Gillett Sweep', W2)}"
                  f"|{cell('Enhanced', W3)}"
                  f"|{cell('Gap Gilt%', W4)}"
                  f"|{cell('Gap Enh%', W5)}|\n")
        out.write(sep())

        sum_opt  = sum_gilt = sum_enh  = 0.0
        cnt_opt  = cnt_gilt = cnt_enh  = 0
        enh_better = gilt_better = tie = 0

        for name in instances:
            opt  = read_cost(name + ".sol")
            gilt = read_cost(name + "_sweep_gillett.txt")
            enh  = read_cost(name + "_enhanced.txt")

            gap_gilt = (gilt - opt) / opt * 100.0 if opt > 0 and gilt > 0 else -1.0
            gap_enh  = (enh  - opt) / opt * 100.0 if opt > 0 and enh  > 0 else -1.0

            out.write(f"|{cell(name, W0)}"
                      f"|{cell_d(opt,      W1)}"
                      f"|{cell_d(gilt,     W2)}"
                      f"|{cell_d(enh,      W3)}"
                      f"|{cell_d(gap_gilt, W4)}"
                      f"|{cell_d(gap_enh,  W5)}|\n")

            if opt  > 0: sum_opt  += opt;  cnt_opt  += 1
            if gilt > 0: sum_gilt += gilt; cnt_gilt += 1
            if enh  > 0: sum_enh  += enh;  cnt_enh  += 1

            if gilt > 0 and enh > 0:
                if   enh  < gilt - 1e-6: enh_better  += 1
                elif gilt < enh  - 1e-6: gilt_better += 1
                else:                    tie         += 1

        # Average row
        avg_opt      = sum_opt  / cnt_opt  if cnt_opt  > 0 else -1.0
        avg_gilt     = sum_gilt / cnt_gilt if cnt_gilt > 0 else -1.0
        avg_enh      = sum_enh  / cnt_enh  if cnt_enh  > 0 else -1.0
        avg_gap_gilt = (avg_gilt - avg_opt) / avg_opt * 100 if avg_opt > 0 and avg_gilt > 0 else -1.0
        avg_gap_enh  = (avg_enh  - avg_opt) / avg_opt * 100 if avg_opt > 0 and avg_enh  > 0 else -1.0

        out.write(sep())
        out.write(f"|{cell('AVERAGE', W0)}"
                  f"|{cell_d(avg_opt,      W1)}"
                  f"|{cell_d(avg_gilt,     W2)}"
                  f"|{cell_d(avg_enh,      W3)}"
                  f"|{cell_d(avg_gap_gilt, W4)}"
                  f"|{cell_d(avg_gap_enh,  W5)}|\n")
        out.write(sep())

        out.write("\n")
        out.write(f"  Enhanced wins   : {enh_better} instance(s)\n")
        out.write(f"  Gillett wins    : {gilt_better} instance(s)\n")
        out.write(f"  Tied            : {tie} instance(s)\n")
        out.write(f"  Total instances : {len(instances)}\n")

    print(f"Done. Results written to {out_path}")


if __name__ == "__main__":
    main()