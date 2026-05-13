#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path


def read_summary(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["x"] = float(row["seq_len_or_tokens"])
            row["layer"] = int(row["layer"])
            row["avg"] = float(row["avg"])
            row["stddev"] = float(row["stddev"])
            row["n"] = int(row["n"])
            rows.append(row)
    return rows


def nice_max(value):
    if value <= 0:
        return 1.0
    exp = math.floor(math.log10(value))
    base = value / (10 ** exp)
    if base <= 2:
        nice = 2
    elif base <= 5:
        nice = 5
    else:
        nice = 10
    return nice * (10 ** exp)


def write_svg(path, title, series, x_label, y_label):
    width, height = 900, 520
    left, right, top, bottom = 78, 28, 54, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = ["#0f766e", "#b45309", "#2563eb", "#be123c", "#4d7c0f", "#7c3aed", "#334155"]

    points = [p for _, pts in series for p in pts]
    if not points:
        return

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    if x_min == x_max:
        x_min = 0
    y_max = nice_max(max(ys))

    def sx(x):
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y):
        return top + plot_h - y / y_max * plot_h

    with open(path, "w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n')
        f.write('<rect width="100%" height="100%" fill="#ffffff"/>\n')
        f.write(f'<text x="{left}" y="32" font-family="Arial" font-size="22" font-weight="700" fill="#111827">{title}</text>\n')
        f.write(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>\n')
        f.write(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>\n')

        for i in range(6):
            y = y_max * i / 5
            yy = sy(y)
            f.write(f'<line x1="{left}" y1="{yy:.1f}" x2="{left + plot_w}" y2="{yy:.1f}" stroke="#e5e7eb"/>\n')
            f.write(f'<text x="{left - 10}" y="{yy + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="#374151">{y:.2g}</text>\n')

        for x in sorted(set(xs)):
            xx = sx(x)
            f.write(f'<line x1="{xx:.1f}" y1="{top + plot_h}" x2="{xx:.1f}" y2="{top + plot_h + 5}" stroke="#111827"/>\n')
            f.write(f'<text x="{xx:.1f}" y="{top + plot_h + 23}" text-anchor="middle" font-family="Arial" font-size="12" fill="#374151">{int(x)}</text>\n')

        for idx, (name, pts) in enumerate(series):
            color = colors[idx % len(colors)]
            pts = sorted(pts)
            path_data = " ".join(("M" if i == 0 else "L") + f"{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(pts))
            f.write(f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="2.5"/>\n')
            for x, y in pts:
                f.write(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5" fill="{color}"/>\n')
            lx = left + 16 + (idx % 3) * 250
            ly = top + plot_h + 48 + (idx // 3) * 18
            f.write(f'<line x1="{lx}" y1="{ly - 4}" x2="{lx + 22}" y2="{ly - 4}" stroke="{color}" stroke-width="2.5"/>\n')
            f.write(f'<text x="{lx + 30}" y="{ly}" font-family="Arial" font-size="12" fill="#111827">{name}</text>\n')

        f.write(f'<text x="{left + plot_w / 2}" y="{height - 12}" text-anchor="middle" font-family="Arial" font-size="13" fill="#111827">{x_label}</text>\n')
        f.write(f'<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})" text-anchor="middle" font-family="Arial" font-size="13" fill="#111827">{y_label}</text>\n')
        f.write("</svg>\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="Path to summary.csv")
    parser.add_argument("--out-dir", required=True, help="Directory for SVG plots")
    args = parser.parse_args()

    rows = read_summary(args.summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for phase in ["prefill", "decode"]:
        attention = [r for r in rows if r["kind"] == "attention_layer" and r["phase"] == phase and r["metric"] == "time_us"]
        if attention:
            modes = sorted(set(r["mode"] for r in attention))
            series = []
            for mode in modes:
                xs = sorted(set(r["x"] for r in attention if r["mode"] == mode))
                pts = []
                for x in xs:
                    vals = [r["avg"] / 1000.0 for r in attention if r["mode"] == mode and r["x"] == x]
                    if vals:
                        pts.append((x, sum(vals) / len(vals)))
                series.append((mode, pts))
            write_svg(out_dir / f"attention_{phase}.svg", f"Attention {phase}", series, "sequence length", "avg layer time (ms)")

    copy_rows = [r for r in rows if r["kind"] == "moe_copy_runtime" and r["metric"] == "time_us"]
    if copy_rows:
        series = []
        for phase in sorted(set(r["phase"] for r in copy_rows)):
            pts = [(r["x"], r["avg"] / 1000.0) for r in copy_rows if r["phase"] == phase]
            series.append((phase, pts))
        write_svg(out_dir / "moe_copy_runtime.svg", "Runtime MoE Expert Copy", series, "sequence length", "time (ms)")

    gemm_rows = [r for r in rows if r["kind"] in ("moe_group_gemm", "moe_serial_gemm") and r["metric"] == "time_us"]
    if gemm_rows:
        for phase in sorted(set(r["phase"] for r in gemm_rows)):
            series = []
            for kind in ["moe_group_gemm", "moe_serial_gemm"]:
                pts = [(r["x"], r["avg"] / 1000.0) for r in gemm_rows if r["kind"] == kind and r["phase"] == phase]
                if pts:
                    series.append((kind, pts))
            write_svg(out_dir / f"moe_gemm_{phase}.svg", f"MoE GEMM {phase}", series, "tokens", "time (ms)")


if __name__ == "__main__":
    main()
