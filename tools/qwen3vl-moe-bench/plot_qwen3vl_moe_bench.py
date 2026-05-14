#!/usr/bin/env python3

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path


COLORS = [
    "#0f766e",
    "#b45309",
    "#2563eb",
    "#be123c",
    "#4d7c0f",
    "#7c3aed",
    "#334155",
    "#0891b2",
    "#c2410c",
    "#4338ca",
]

ATTENTION_METHODS = [
    ("kv_gpu_attn_gpu", "KV GPU + Attn GPU"),
    ("kv_cpu_attn_cpu", "KV CPU + Attn CPU"),
    ("kv_cpu_attn_gpu", "KV CPU + Attn GPU"),
]


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


def fmt_x(x):
    if x >= 1024 and x % 1024 == 0:
        return f"{int(x // 1024)}K"
    return str(int(x))


def line_svg(f, x1, y1, x2, y2, color="#111827", width=1.0):
    f.write(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"/>\n')


def text_svg(f, x, y, text, size=12, anchor="middle", weight="400", color="#111827", rotate=None):
    transform = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
    f.write(
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="Arial" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}"{transform}>{html.escape(text)}</text>\n'
    )


def plot_lines(path, title, series, x_label, y_label):
    width, height = 900, 520
    left, right, top, bottom = 78, 28, 54, 88
    plot_w = width - left - right
    plot_h = height - top - bottom
    points = [p for _, pts in series for p in pts]
    if not points:
        return

    xs = sorted(set(p[0] for p in points))
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
        text_svg(f, left, 32, title, size=22, anchor="start", weight="700")
        line_svg(f, left, top + plot_h, left + plot_w, top + plot_h)
        line_svg(f, left, top, left, top + plot_h)

        for i in range(6):
            y = y_max * i / 5
            yy = sy(y)
            line_svg(f, left, yy, left + plot_w, yy, color="#e5e7eb")
            text_svg(f, left - 10, yy + 4, f"{y:.2g}", anchor="end", color="#374151")

        for x in xs:
            xx = sx(x)
            line_svg(f, xx, top + plot_h, xx, top + plot_h + 5)
            text_svg(f, xx, top + plot_h + 23, fmt_x(x), color="#374151")

        for idx, (name, pts) in enumerate(series):
            color = COLORS[idx % len(COLORS)]
            pts = sorted(pts)
            path_data = " ".join(("M" if i == 0 else "L") + f"{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(pts))
            f.write(f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="2.5"/>\n')
            for x, y in pts:
                f.write(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5" fill="{color}"/>\n')
            lx = left + 16 + (idx % 3) * 250
            ly = top + plot_h + 50 + (idx // 3) * 18
            line_svg(f, lx, ly - 4, lx + 22, ly - 4, color=color, width=2.5)
            text_svg(f, lx + 30, ly, name, anchor="start")

        text_svg(f, left + plot_w / 2, height - 14, x_label, size=13)
        text_svg(f, 18, top + plot_h / 2, y_label, size=13, rotate=-90)
        f.write("</svg>\n")


def plot_attention_panels(path, attn):
    width, height = 1120, 560
    left, right, top, bottom = 78, 28, 72, 116
    gap = 58
    panel_w = (width - left - right - gap) / 2
    panel_h = height - top - bottom
    phases = ["prefill", "decode"]
    points = [
        (x, y)
        for (_, phase, x), y in attn.items()
        if phase in phases
    ]
    if not points:
        return

    xs = sorted(set(x for x, _ in points))
    x_min, x_max = min(xs), max(xs)
    if x_min == x_max:
        x_min = 0
    y_max_by_phase = {}
    for phase in phases:
        phase_ys = [y for (_, p, _), y in attn.items() if p == phase]
        if phase_ys:
            y_max_by_phase[phase] = nice_max(max(phase_ys))

    def sx(x, panel_idx):
        px = left + panel_idx * (panel_w + gap)
        return px + (x - x_min) / (x_max - x_min) * panel_w

    def sy(y, phase):
        return top + panel_h - y / y_max_by_phase[phase] * panel_h

    with open(path, "w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n')
        f.write('<rect width="100%" height="100%" fill="#ffffff"/>\n')
        text_svg(f, left, 34, "Attention Time", size=22, anchor="start", weight="700")

        for pidx, phase in enumerate(phases):
            px = left + pidx * (panel_w + gap)
            text_svg(f, px + panel_w / 2, top - 18, phase.title(), size=15, weight="700")
            line_svg(f, px, top + panel_h, px + panel_w, top + panel_h)
            line_svg(f, px, top, px, top + panel_h)
            for i in range(6):
                y = y_max_by_phase[phase] * i / 5
                yy = sy(y, phase)
                line_svg(f, px, yy, px + panel_w, yy, color="#e5e7eb")
                text_svg(f, px - 10, yy + 4, f"{y:.2g}", anchor="end", color="#374151")
            for x in xs:
                xx = sx(x, pidx)
                line_svg(f, xx, top + panel_h, xx, top + panel_h + 5)
                text_svg(f, xx, top + panel_h + 23, fmt_x(x), color="#374151")
            for midx, (method, label) in enumerate(ATTENTION_METHODS):
                pts = sorted((x, y) for (m, p, x), y in attn.items() if m == method and p == phase)
                if not pts:
                    continue
                color = COLORS[midx % len(COLORS)]
                path_data = " ".join(("M" if i == 0 else "L") + f"{sx(x, pidx):.1f},{sy(y, phase):.1f}" for i, (x, y) in enumerate(pts))
                f.write(f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="2.5"/>\n')
                for x, y in pts:
                    f.write(f'<circle cx="{sx(x, pidx):.1f}" cy="{sy(y, phase):.1f}" r="3.5" fill="{color}"/>\n')

        for idx, (_, label) in enumerate(ATTENTION_METHODS):
            color = COLORS[idx % len(COLORS)]
            lx = left + 18 + idx * 300
            ly = top + panel_h + 58
            line_svg(f, lx, ly - 4, lx + 24, ly - 4, color=color, width=2.5)
            text_svg(f, lx + 32, ly, label, anchor="start")

        text_svg(f, left + (width - left - right) / 2, height - 18, "sequence length", size=13)
        text_svg(f, 18, top + panel_h / 2, "avg layer time (ms)", size=13, rotate=-90)
        f.write("</svg>\n")


def plot_ratio_panels(path, ratios):
    width, height = 1120, 520
    left, right, top, bottom = 78, 28, 72, 88
    gap = 58
    panel_w = (width - left - right - gap) / 2
    panel_h = height - top - bottom
    phases = ["prefill", "decode"]
    points = [
        (x, y)
        for (phase, x), y in ratios.items()
        if phase in phases
    ]
    if not points:
        return

    xs = sorted(set(x for x, _ in points))
    x_min, x_max = min(xs), max(xs)
    if x_min == x_max:
        x_min = 0
    y_max_by_phase = {}
    for phase in phases:
        phase_ys = [y for (p, _), y in ratios.items() if p == phase]
        if phase_ys:
            y_max_by_phase[phase] = nice_max(max(phase_ys))

    def sx(x, panel_idx):
        px = left + panel_idx * (panel_w + gap)
        return px + (x - x_min) / (x_max - x_min) * panel_w

    def sy(y, phase):
        return top + panel_h - y / y_max_by_phase[phase] * panel_h

    with open(path, "w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n')
        f.write('<rect width="100%" height="100%" fill="#ffffff"/>\n')
        text_svg(f, left, 34, "Attention / Expert H2D Copy", size=22, anchor="start", weight="700")

        for pidx, phase in enumerate(phases):
            if phase not in y_max_by_phase:
                continue
            px = left + pidx * (panel_w + gap)
            text_svg(f, px + panel_w / 2, top - 18, phase.title(), size=15, weight="700")
            line_svg(f, px, top + panel_h, px + panel_w, top + panel_h)
            line_svg(f, px, top, px, top + panel_h)
            for i in range(6):
                y = y_max_by_phase[phase] * i / 5
                yy = sy(y, phase)
                line_svg(f, px, yy, px + panel_w, yy, color="#e5e7eb")
                text_svg(f, px - 10, yy + 4, f"{y:.2g}", anchor="end", color="#374151")
            for x in xs:
                xx = sx(x, pidx)
                line_svg(f, xx, top + panel_h, xx, top + panel_h + 5)
                text_svg(f, xx, top + panel_h + 23, fmt_x(x), color="#374151")
            pts = sorted((x, y) for (p, x), y in ratios.items() if p == phase)
            if pts:
                color = COLORS[1]
                path_data = " ".join(("M" if i == 0 else "L") + f"{sx(x, pidx):.1f},{sy(y, phase):.1f}" for i, (x, y) in enumerate(pts))
                f.write(f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="2.5"/>\n')
                for x, y in pts:
                    f.write(f'<circle cx="{sx(x, pidx):.1f}" cy="{sy(y, phase):.1f}" r="3.5" fill="{color}"/>\n')

        lx = left + 18
        ly = top + panel_h + 56
        line_svg(f, lx, ly - 4, lx + 24, ly - 4, color=COLORS[1], width=2.5)
        text_svg(f, lx + 32, ly, "KV CPU + Attn CPU / one expert H2D", anchor="start")
        text_svg(f, left + (width - left - right) / 2, height - 18, "sequence length", size=13)
        text_svg(f, 18, top + panel_h / 2, "ratio", size=13, rotate=-90)
        f.write("</svg>\n")


def plot_moe_gemm(path, panels):
    width, height = 1120, 620
    left, right, top, bottom = 72, 28, 70, 124
    gap = 54
    panel_w = (width - left - right - gap) / 2
    panel_h = height - top - bottom
    all_points = [p for _, series in panels for _, pts in series for p in pts]
    if not all_points:
        return

    xs = sorted(set(p[0] for p in all_points))
    x_min, x_max = min(xs), max(xs)
    if x_min == x_max:
        x_min = 0
    y_max = nice_max(max(p[1] for p in all_points))

    def sx(x, panel_idx):
        px = left + panel_idx * (panel_w + gap)
        return px + (x - x_min) / (x_max - x_min) * panel_w

    def sy(y):
        return top + panel_h - y / y_max * panel_h

    with open(path, "w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n')
        f.write('<rect width="100%" height="100%" fill="#ffffff"/>\n')
        text_svg(f, left, 34, "MoE GEMM FFN", size=22, anchor="start", weight="700")

        for pidx, (panel_name, series) in enumerate(panels):
            px = left + pidx * (panel_w + gap)
            text_svg(f, px + panel_w / 2, top - 18, panel_name.title(), size=15, weight="700")
            line_svg(f, px, top + panel_h, px + panel_w, top + panel_h)
            line_svg(f, px, top, px, top + panel_h)
            for i in range(6):
                y = y_max * i / 5
                yy = sy(y)
                line_svg(f, px, yy, px + panel_w, yy, color="#e5e7eb")
                text_svg(f, px - 10, yy + 4, f"{y:.2g}", anchor="end", color="#374151")
            for x in xs:
                xx = sx(x, pidx)
                line_svg(f, xx, top + panel_h, xx, top + panel_h + 5)
                text_svg(f, xx, top + panel_h + 23, fmt_x(x), color="#374151")
            for idx, (name, pts) in enumerate(series):
                color = COLORS[idx % len(COLORS)]
                dash = ' stroke-dasharray="5 4"' if name.startswith("serial") else ""
                pts = sorted(pts)
                path_data = " ".join(("M" if i == 0 else "L") + f"{sx(x, pidx):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(pts))
                f.write(f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="2.0"{dash}/>\n')

        hs = sorted({
            int(name.rsplit("=", 1)[1])
            for _, series in panels
            for name, _ in series
            if "=" in name
        })
        legend_items = []
        for h in hs:
            legend_items.append((f"group h={h}", COLORS[(h - 1) % len(COLORS)], False))
        for h in hs:
            legend_items.append((f"serial h={h}", COLORS[(h - 1) % len(COLORS)], True))
        for idx, (name, color, dashed) in enumerate(legend_items):
            lx = left + 10 + (idx % 4) * 250
            ly = top + panel_h + 56 + (idx // 4) * 18
            dash = ' stroke-dasharray="5 4"' if dashed else ""
            f.write(f'<line x1="{lx:.1f}" y1="{ly - 4:.1f}" x2="{lx + 22:.1f}" y2="{ly - 4:.1f}" stroke="{color}" stroke-width="2.0"{dash}/>\n')
            text_svg(f, lx + 30, ly, name, anchor="start")

        text_svg(f, left + (width - left - right) / 2, height - 14, "tokens", size=13)
        text_svg(f, 18, top + panel_h / 2, "time (ms)", size=13, rotate=-90)
        f.write("</svg>\n")


def average_attention(rows):
    grouped = defaultdict(list)
    for r in rows:
        if r["kind"] == "attention_layer" and r["metric"] == "time_us":
            mode = attention_method(r["mode"])
            if mode:
                grouped[(mode, r["phase"], r["x"])].append(r["avg"] / 1000.0)
    return {k: sum(v) / len(v) for k, v in grouped.items() if v}


def average_expert_copy_ms(rows):
    copy_times = [
        r["avg"] / 1000.0
        for r in rows
        if r["kind"] == "expert_h2d_pinned" and r["metric"] == "time_us"
    ]
    if not copy_times:
        return None
    return sum(copy_times) / len(copy_times)


def attention_method(mode):
    prefix = "moe_cpu_offload_"
    if mode.startswith(prefix):
        mode = mode[len(prefix):]
    if mode in {method for method, _ in ATTENTION_METHODS}:
        return mode
    return None


def attention_label(mode, phase):
    labels = dict(ATTENTION_METHODS)
    return f"{labels.get(mode, mode)} {phase}"


def micro_h(row):
    mode = row["mode"]
    if mode == "micro":
        return 1
    if mode.startswith("micro_h"):
        try:
            return int(mode.removeprefix("micro_h"))
        except ValueError:
            return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="Path to summary.csv")
    parser.add_argument("--expert-summary", help="Optional summary.csv that contains expert_h2d_pinned rows")
    parser.add_argument("--out-dir", required=True, help="Directory for SVG plots")
    args = parser.parse_args()

    rows = read_summary(args.summary)
    expert_rows = rows
    if args.expert_summary:
        expert_rows = read_summary(args.expert_summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    attn = average_attention(rows)
    plot_attention_panels(out_dir / "figure1_attention_time.svg", attn)

    expert_copy_ms = average_expert_copy_ms(expert_rows)
    if expert_copy_ms and expert_copy_ms > 0:
        ratio = {
            (phase, x): y / expert_copy_ms
            for (mode, phase, x), y in attn.items()
            if mode == "kv_cpu_attn_cpu"
        }
        plot_ratio_panels(out_dir / "figure2_attention_expert_copy_ratio.svg", ratio)

    panels = []
    for phase in ["prefill", "decode"]:
        series = []
        hs = sorted({
            h
            for h in (micro_h(r) for r in rows)
            if h is not None
        })
        for h in hs:
            for kind, label in [("moe_group_gemm", f"group h={h}"), ("moe_serial_gemm", f"serial h={h}")]:
                pts = [
                    (r["x"], r["avg"] / 1000.0)
                    for r in rows
                    if r["kind"] == kind and micro_h(r) == h and r["phase"] == phase and r["metric"] == "time_us"
                ]
                if pts:
                    series.append((label, pts))
        if series:
            panels.append((phase, series))
    plot_moe_gemm(out_dir / "figure3_moe_gemm.svg", panels)


if __name__ == "__main__":
    main()
