"""Render Fig. 2-4 of the paper in Indonesian (id) and English (en).

Fig. 2 reads the desktop test metrics in Training/Output/results/, Fig. 3 and Fig. 4 read the
per-attempt log of the Poco M4 Pro in this folder.

Usage: python make_figures.py [POCO_CSV] [OUT_DIR]
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

CNN_C = "#2a78d6"
MVIT_C = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "Training", "Output", "results")
METRIC_KEYS = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
# Class groups as consecutive blocks of the attempt log (5 attempts per class, label order)
GROUP_BOUNDS = [(0, 60), (60, 90), (90, 110), (110, 140), (140, 160)]

TEXT = {
    "id": {
        "metrics": ["Akurasi", "Presisi\n(tertimbang)", "Recall\n(tertimbang)", "F1\n(tertimbang)"],
        "y_desk": "Nilai pada data uji desktop (%)",
        "x_lat": "Percobaan (5 per kelas, kelas urut label)",
        "y_lat": "Latensi inferensi per prediksi (s)",
        "day1": "hari 1", "day2": "hari 2",
        "groups": ["Kata\nsehari-hari\n(60)", "Kata\ntanya\n(30)", "Warna\n(20)", "Kata kerja\ndan benda\n(30)",
                   "Waktu\n(20)"],
        "y_grp": "Akurasi pada Poco M4 Pro (%)",
        "dec": ",",
    },
    "en": {
        "metrics": ["Accuracy", "Precision\n(weighted)", "Recall\n(weighted)", "F1\n(weighted)"],
        "y_desk": "Desktop test-set score (%)",
        "x_lat": "Attempt (5 per class, classes in label order)",
        "y_lat": "Inference latency per prediction (s)",
        "day1": "day 1", "day2": "day 2",
        "groups": ["Everyday\nwords (60)", "Question\nwords (30)", "Colors\n(20)", "Verbs and\nnouns (30)",
                   "Times of\nday (20)"],
        "y_grp": "Accuracy on Poco M4 Pro (%)",
        "dec": ".",
    },
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "axes.linewidth": 0.6,
})


def style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def legend(ax, ncol=2):
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=ncol, frameon=False, fontsize=7,
                    handlelength=1.4, columnspacing=1.5)
    for t in leg.get_texts():
        t.set_color(INK)


def grouped_bars(ax, labels, a, b, decimals, dec, label_size=6.0):
    w, gap = 0.34, 0.02
    xs = range(len(labels))
    b1 = ax.bar([i - w / 2 - gap / 2 for i in xs], a, width=w, color=CNN_C, label="CNNsa-BiLSTM")
    b2 = ax.bar([i + w / 2 + gap / 2 for i in xs], b, width=w, color=MVIT_C, label="MViTv2")
    for bars, vals in ((b1, a), (b2, b)):
        for bar, v in zip(bars, vals):
            s = ("%%.%df" % decimals) % v
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, s.replace(".", dec), ha="center", va="bottom",
                    fontsize=label_size, color=INK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])


def desktop_metrics(name):
    d = json.load(open(os.path.join(RESULTS, name), encoding="utf-8"))
    return [round(d[k] * 100, 2) for k in METRIC_KEYS]


def group_accuracy(rows, model):
    acc = []
    for s, e in GROUP_BOUNDS:
        g = rows[s:e]
        acc.append(round(100 * sum(r[model + "Correct"] == "true" for r in g) / len(g), 1))
    return acc


def fig2(lang, out):
    t = TEXT[lang]
    cnn = desktop_metrics("evaluation_summary.json")
    mvit = desktop_metrics("transformer_evaluation_summary.json")
    fig, ax = plt.subplots(figsize=(3.3, 2.0), dpi=300)
    style(ax)
    grouped_bars(ax, t["metrics"], cnn, mvit, 2, t["dec"], label_size=5.2)
    ax.set_ylabel(t["y_desk"])
    legend(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(out, "fig2_desktop_%s.png" % lang))
    plt.close(fig)


def fig3(lang, out, rows):
    t = TEXT[lang]
    x = [int(r["attempt"]) for r in rows]
    cnn = [int(r["cnnLatencyMs"]) / 1000 for r in rows]
    mvit = [int(r["mvitLatencyMs"]) / 1000 for r in rows]
    fig, ax = plt.subplots(figsize=(3.3, 2.1), dpi=300)
    style(ax)
    ax.set_yscale("log")
    ax.plot(x, cnn, color=CNN_C, linewidth=0.9, marker="o", markersize=2.2, markeredgewidth=0, label="CNNsa-BiLSTM")
    ax.plot(x, mvit, color=MVIT_C, linewidth=0.9, marker="o", markersize=2.2, markeredgewidth=0, label="MViTv2")
    ax.set_ylim(0.1, 400)
    ticks = [0.1, 0.3, 1, 3, 10, 30, 100, 300]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter([("%g" % v).replace(".", t["dec"]) for v in ticks]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(0, 161)
    ax.set_xticks([1, 20, 40, 60, 80, 100, 120, 140, 160])
    ax.set_xlabel(t["x_lat"])
    ax.set_ylabel(t["y_lat"])
    ax.axvline(100.5, color=AXIS, linewidth=0.6)
    ax.text(102, 0.13, t["day2"], color=INK2, fontsize=6.5)
    ax.text(2, 0.13, t["day1"], color=INK2, fontsize=6.5)
    peak = max(range(len(mvit)), key=lambda i: mvit[i])
    ax.annotate("%.0f s" % mvit[peak], xy=(x[peak], mvit[peak]), xytext=(x[peak] + 9, mvit[peak] * 0.95),
                fontsize=6.5, color=INK, va="center", arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.5))
    legend(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(out, "fig3_latency_%s.png" % lang))
    plt.close(fig)


def fig4(lang, out, rows):
    t = TEXT[lang]
    fig, ax = plt.subplots(figsize=(3.3, 2.0), dpi=300)
    style(ax)
    grouped_bars(ax, t["groups"], group_accuracy(rows, "cnn"), group_accuracy(rows, "mvit"), 0, t["dec"])
    ax.set_ylabel(t["y_grp"])
    legend(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(out, "fig4_groups_%s.png" % lang))
    plt.close(fig)


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "poco_m4_pro_attempts.csv")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "charts")
    os.makedirs(out_dir, exist_ok=True)
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    for lang in ("id", "en"):
        fig2(lang, out_dir)
        fig3(lang, out_dir, rows)
        fig4(lang, out_dir, rows)
    print("figures written to", out_dir)
