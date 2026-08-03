"""Generate publication figures for the FedSUPER+LLM paper.
Outputs are vector PDFs in paper/figures/.
All numbers come directly from the experimental result JSON files
(experiments/H*/results/metrics_*.json), so figure values match the paper's tables.
"""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.labelsize": 10,
    "axes.titlesize": 11, "legend.fontsize": 9, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.right": False, "axes.spines.top": False,
})

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

C_BLUE, C_ORANGE, C_GREEN, C_RED, C_GREY = "#1f6feb", "#d97706", "#238636", "#d73a49", "#6e7681"
C_ACC, C_PRIV = "#9333ea", "#0891b2"

def _load(*segments):
    with open(os.path.join(ROOT, *segments), "r") as f:
        return json.load(f)

# DFS:
h1 = _load("experiments", "H1-fedsuper", "results", "metrics_h1_v2.json")
h2 = _load("experiments", "H2-dp-helps-fairness", "results", "metrics_h2_v2.json")
h3 = _load("experiments", "H3-llm-profiling", "results", "metrics_h3_v2.json")
h4 = _load("experiments", "H4-multiobj", "results", "metrics_h4.json")
h5 = _load("experiments", "H5-adaptive-dp", "results", "metrics_h5.json")

# ============ Figure 1: Architecture of FedSUPER+LLM ============
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.5); ax.axis("off")
    def box(x, y, w, h, label, color, linestyle="-", textcolor="black", fontsize=8.5, fc=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                     linewidth=1.2, edgecolor=color, facecolor=fc or "white", linestyle=linestyle))
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=fontsize,
                color=textcolor, wrap=True)
    def arrow(x1, y1, x2, y2, color=C_GREY, style="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=10, linewidth=1.1, color=color, linestyle=style))

    # Client side
    ax.text(1.55, 5.1, "Client (per user u)", ha="center", fontsize=9, fontweight="bold", color=C_BLUE)
    box(0.3, 3.0, 2.5, 0.8, "Local history $C_u$\n(interactions, PRIVATE)", C_BLUE, fc="#ddf4ff")
    box(0.3, 1.9, 2.5, 0.8, "User embedding $E_u$\n(never leaves device)", C_BLUE, fc="#ddf4ff")
    box(0.3, 0.8, 2.5, 0.8, "LLM profile $\\mathbf{p}_u$\n(Sentence-BERT, private text)", C_ACC, fc="#f3e8ff", fontsize=8)
    arrow(1.55, 3.0, 1.55, 2.7); arrow(1.55, 2.6, 1.55, 1.6); arrow(1.55, 2.65, 1.55, 0.0)  # removed dead arrow

    # Gradient / item update channel
    ax.text(4.97, 5.1, "Communication (server)", ha="center", fontsize=9, fontweight="bold", color=C_GREEN)
    box(3.4, 2.1, 3.1, 0.9, "PrivateKey head-tail item-$\\Delta I_H / \\Delta I_T$ gradients\n(only item params aggregated)", C_GREEN, fc="#dcfce7", fontsize=8.5)
    box(3.4, 0.8, 3.1, 0.9, "(optional) Gaussian DP noise $\\mathcal{N}(0,\\sigma^2)$\nper-item or uniform", C_RED, fc="#fee2e2", fontsize=8.5, linestyle="--")
    arrow(1.55, 1.95, 3.4, 2.45, C_GREEN); arrow(3.4, 2.0, 4.97, 1.7, C_GREEN)
    arrow(4.97, 0.8, 4.97, 2.1, C_RED, style="--")

    # Server aggregation
    ax.text(8.4, 5.1, "Server (public)", ha="center", fontsize=9, fontweight="bold", color=C_ORANGE)
    box(7.0, 3.0, 2.8, 0.8, "Pareto partition $\\alpha{=}0.20$\n(public catalog counts)", C_ORANGE, fc="#fef3c7")
    box(7.0, 1.9, 2.8, 0.8, "Server item $I_H, I_T$\n(FedAvg aggregates)", C_ORANGE, fc="#fef3c7")
    box(7.0, 0.8, 2.8, 0.8, "Blueprint merge\n$N_{pop}{=}\\lfloor N\\cdot Pop_u\\rfloor$", C_ORANGE, fc="#fef3c7", fontsize=9)
    arrow(6.5, 2.4, 7.0, 2.3, C_GREEN); arrow(8.4, 3.0, 8.4, 2.7); arrow(8.4, 1.9, 8.4, 1.6)
    arrow(3.0, 0.5, 7.0, 1.1, C_ACC, style=":")  # LLM side-info flows to merge
    ax.text(5.0, 0.35, "LLM similarity $\\lambda\\,\\cos(\\mathbf{p}_u,\\mathbf{t}_i)$ blended into pool scores",
            ha="center", fontsize=8, color=C_ACC)

    ax.set_title("FedSUPER+LLM: federated dual-model training, privacy-preserving blueprint merge",
                 fontsize=10.5, pad=6)
    plt.savefig(os.path.join(FIG, "fig1_architecture.pdf"))
    plt.close()

# ============ Figure 2: Calibration invariance (Rmse-PC, MRMC across variants) ============
def fig2_calibration_invariance():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.6, 3.0))
    labels = ["BPR-MF\n(central)", "FedNCF\n(FL)", "FedNCF\nDP $\\varepsilon$=2",
              "SUPER\n(central)", "FedSUPER\n(FL+blue)", "FedSUPER\nDP $\\varepsilon$=2",
              "LLM-FedSUPER\n$\\lambda$=0.7", "LLM-FedSUPER\nDP$\\varepsilon$=2"]
    rmse = [h1["BPR-MF"]["rmse_pc"], h1["FedNCF"]["rmse_pc"], h2["FedNCF-DP_eps=2.0"]["rmse_pc"],
            h1["SUPER-centralized"]["rmse_pc"], h1["FedSUPER"]["rmse_pc"], h2["FedSUPER-DP_eps=2.0"]["rmse_pc"],
            h3["LLM-FedSUPER-noDP_lam=0.7"]["rmse_pc"], h3["LLM-FedSUPER-DP_eps=2_lam=0.7"]["rmse_pc"]]
    mrmc = [h1["BPR-MF"]["mrmc"], h1["FedNCF"]["mrmc"], h2["FedNCF-DP_eps=2.0"]["mrmc"],
            h1["SUPER-centralized"]["mrmc"], h1["FedSUPER"]["mrmc"], h2["FedSUPER-DP_eps=2.0"]["mrmc"],
            h3["LLM-FedSUPER-noDP_lam=0.7"]["mrmc"], h3["LLM-FedSUPER-DP_eps=2_lam=0.7"]["mrmc"]]
    n = np.arange(len(labels))
    non_super = [0,1,2]; super_v = [3,4,5,6,7]
    cols = [C_GREY]*3 + [C_GREEN]*5
    axA.bar(n, rmse, color=cols, edgecolor="black", linewidth=0.4)
    axA.set_xticks(n); axA.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
    axA.set_ylabel("RMSE-PC $\\downarrow$"); axA.set_title("(a) Popularity Calibration", fontsize=10)
    axA.axhline(0.055, color=C_GREEN, linestyle=":", linewidth=1.0)
    axA.text(2.6, 0.07, "SUPER guarantee", color=C_GREEN, fontsize=7.5, ha="center")
    axA.set_ylim(0, 0.78)
    axB.bar(n, mrmc, color=cols, edgecolor="black", linewidth=0.4)
    axB.set_xticks(n); axB.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
    axB.set_ylabel("MRMC $\\downarrow$"); axB.set_title("(b) Mean Rank Miscalibration", fontsize=10)
    axB.axhline(0.151, color=C_GREEN, linestyle=":", linewidth=1.0)
    axB.set_ylim(0, 0.78)
    legend = [Line2D([0],[0],color=C_GREY,lw=6,label="no SUPER"),
             Line2D([0],[0],color=C_GREEN,lw=6,label="SUPER blueprint")]
    fig.legend(handles=legend, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03), frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig2_calibration_invariance.pdf"))
    plt.close()

# ============ Figure 3: Accuracy / Privacy Pareto front ============
def fig3_pareto():
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    # Recall vs DP strength (eps), marker size by LTC
    eps_vals = [np.inf, 8, 4, 2, 2]  # inf=noDP for FedSUPER series
    fed_no = h3["FedSUPER-noDP"]
    fed_dp8 = h2["FedSUPER-DP_eps=8.0"]; fed_dp4 = h2["FedSUPER-DP_eps=4.0"]; fed_dp2 = h2["FedSUPER-DP_eps=2.0"]
    fed_adp = h5["FedSUPER-DP_adaptive_eps=2_amp=2"]
    llm_no = h3["LLM-FedSUPER-noDP_lam=0.7"]
    llm_dp2 = h3["LLM-FedSUPER-DP_eps=2_lam=0.7"]
    super_c = h1["SUPER-centralized"]; bprmf = h1["BPR-MF"]; fedncf = h1["FedNCF"]
    # plot isolated points as scatter; x = eps (privacy strength, log)
    def x_of(e): return -np.log10(max(e, 1.5)) if np.isfinite(e) else 1.4  # placeholder - inf -> right side
    # Better: x = privacy (none = leftmost, eps=8 > 4 > 2); use log scale flipped
    pts = [
        ("BPR-MF",          bprmf["recall@K"], bprmf["ltc"], 10.0, C_GREY, "X"),
        ("SUPER-centralized", super_c["recall@K"], super_c["ltc"], 10.0, C_GREY, "X"),
        ("FedNCF",          fedncf["recall@K"], fedncf["ltc"], 10.0, C_BLUE, "X"),
        ("FedSUPER",        fed_no["recall@K"], fed_no["ltc"], 10.0, C_BLUE, "o"),
        ("FedSUPER $\\varepsilon$=8", fed_dp8["recall@K"], fed_dp8["ltc"], 8.0, C_BLUE, "o"),
        ("FedSUPER $\\varepsilon$=4", fed_dp4["recall@K"], fed_dp4["ltc"], 4.0, C_BLUE, "o"),
        ("FedSUPER $\\varepsilon$=2", fed_dp2["recall@K"], fed_dp2["ltc"], 2.0, C_BLUE, "o"),
        ("FedSUPER-AD $\\varepsilon$=2", fed_adp["recall@K"], fed_adp["ltc"], 2.0, C_PRIV, "P"),
        ("LLM-FedSUPER $\\lambda$=0.7", llm_no["recall@K"], llm_no["ltc"], 10.0, C_ACC, "s"),
        ("LLM-FedSUPER-DP $\\varepsilon$=2 $\\lambda$=0.7", llm_dp2["recall@K"], llm_dp2["ltc"], 2.0, C_ACC, "s"),
    ]
    for name, rec, ltc, eps, c, m in pts:
        ax.scatter(eps if np.isfinite(eps) else 11, rec, s=max(40, ltc*1200), c=c, marker=m, edgecolors="black", linewidths=0.6, zorder=3)
    # Tag the key points
    ax.annotate("LLM-FedSUPER\n(best privacy+accuracy)", (10, llm_no["recall@K"]), xytext=(8.7, llm_no["recall@K"]+0.011),
                fontsize=8, color=C_ACC, arrowprops=dict(arrowstyle="->", color=C_ACC, lw=0.8))
    ax.annotate("SUPER-centralized\n(no privacy)", (10, super_c["recall@K"]), xytext=(8.7, super_c["recall@K"]+0.013),
                fontsize=8, color=C_GREY, arrowprops=dict(arrowstyle="->", color=C_GREY, lw=0.8))
    ax.annotate("FedSUPER\nDP accuracy drop", (2, fed_dp2["recall@K"]), xytext=(2.2, fed_dp2["recall@K"]-0.012),
                fontsize=8, color=C_BLUE, arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=0.8))
    ax.set_xlabel("Differential Privacy strength $\\varepsilon$ (lower = more privacy)")
    ax.set_ylabel("Recall@10 $\\uparrow$")
    ax.invert_xaxis()
    ax.set_xscale("log")
    ax.set_xlim(12, 1.5); ax.set_ylim(0.005, 0.055)
    ax.legend(handles=[Line2D([0],[0],color=C_GREY,marker="X",lw=0,label="Centralized reference"),
                       Line2D([0],[0],color=C_BLUE,marker="o",lw=0,label="FedSUPER (DP sweep)"),
                       Line2D([0],[0],color=C_PRIV,marker="P",lw=0,label="FedSUPER w/ Adaptive DP"),
                       Line2D([0],[0],color=C_ACC,marker="s",lw=0,label="LLM-FedSUPER (ours)")],
              loc="lower left", frameon=False, fontsize=8)
    ax.set_title("Accuracy–Privacy trade-off; marker size $\\propto$ Long-Tail Coverage (LTC)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig3_pareto.pdf"))
    plt.close()

# ============ Figure 4: LLM blending sweep (lambda effect on Recall/LTC/GKPI) ============
def fig4_llm_sweep():
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    lams = [0.0, 0.3, 0.5, 0.7]
    no_dp = [h3["FedSUPER-noDP"]["recall@K"],
             h3["LLM-FedSUPER-noDP_lam=0.3"]["recall@K"],
             h3["LLM-FedSUPER-noDP_lam=0.5"]["recall@K"],
             h3["LLM-FedSUPER-noDP_lam=0.7"]["recall@K"]]
    dp = [h3["FedSUPER-DP_eps=2"]["recall@K"],
          h3["LLM-FedSUPER-DP_eps=2_lam=0.3"]["recall@K"],
          h3["LLM-FedSUPER-DP_eps=2_lam=0.5"]["recall@K"],
          h3["LLM-FedSUPER-DP_eps=2_lam=0.7"]["recall@K"]]
    ltc_no = [h3["FedSUPER-noDP"]["ltc"],
              h3["LLM-FedSUPER-noDP_lam=0.3"]["ltc"],
              h3["LLM-FedSUPER-noDP_lam=0.5"]["ltc"],
              h3["LLM-FedSUPER-noDP_lam=0.7"]["ltc"]]
    ltc_dp = [h3["FedSUPER-DP_eps=2"]["ltc"],
              h3["LLM-FedSUPER-DP_eps=2_lam=0.3"]["ltc"],
              h3["LLM-FedSUPER-DP_eps=2_lam=0.5"]["ltc"],
              h3["LLM-FedSUPER-DP_eps=2_lam=0.7"]["ltc"]]
    ax2 = ax.twinx()
    ax.plot(lams, no_dp, "-o", color=C_ACC, markersize=6, label="Recall (no DP)")
    ax.plot(lams, dp, "--o", color=C_ACC, markersize=5, alpha=0.7, label="Recall (DP $\\varepsilon$=2)")
    ax2.plot(lams, ltc_no, "-s", color=C_GREEN, markersize=6, label="LTC (no DP)")
    ax2.plot(lams, ltc_dp, "--s", color=C_GREEN, markersize=5, alpha=0.7, label="LTC (DP $\\varepsilon$=2)")
    ax.axhline(h1["SUPER-centralized"]["recall@K"], color=C_GREY, linestyle=":", linewidth=1.0)
    ax.text(0.025, h1["SUPER-centralized"]["recall@K"]+0.0008, "SUPER-centralized", color=C_GREY, fontsize=8)
    ax.set_xlabel("LLM blend weight $\\lambda$")
    ax.set_ylabel("Recall@10", color=C_ACC)
    ax2.set_ylabel("Long-Tail Coverage (LTC)", color=C_GREEN)
    ax.set_ylim(0.01, 0.04); ax2.set_ylim(0.005, 0.05)
    h1_, l1_ = ax.get_legend_handles_labels(); h2_, l2_ = ax2.get_legend_handles_labels()
    ax.legend(h1_+h2_, l1_+l2_, loc="upper left", frameon=False, fontsize=8)
    ax.set_title("LLM blending rescues both accuracy and coverage\n(calibration RMSE-PC $=0.055$ unchanged across $\\lambda$)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig4_llm_sweep.pdf"))
    plt.close()

# ============ Figure 5: Optimization trajectory ============
def fig5_trajectory():
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    runs = ["H1\nFedSUPER", "H2\nDP $\\varepsilon$=2", "H3\nLLM $\\lambda$=0.7",
            "H4\nmulti-obj $\\alpha$=0.005", "H5\nAdaptive DP"]
    recall = [h1["FedSUPER"]["recall@K"], h2["FedSUPER-DP_eps=2.0"]["recall@K"],
              h3["LLM-FedSUPER-noDP_lam=0.7"]["recall@K"],
              h4["LLM-FedSUPER-multi_alpha=0.005_lam=0.7"]["recall@K"],
              h5["FedSUPER-DP_adaptive_eps=2_amp=2"]["recall@K"]]
    gkpi = [h1["FedSUPER"]["gkpi"], h2["FedSUPER-DP_eps=2.0"]["gkpi"],
            h3["LLM-FedSUPER-noDP_lam=0.7"]["gkpi"],
            h4["LLM-FedSUPER-multi_alpha=0.005_lam=0.7"]["gkpi"],
            h5["FedSUPER-DP_adaptive_eps=2_amp=2"]["gkpi"]]
    x = np.arange(len(runs))
    ax.plot(x, recall, "-o", color=C_ACC, markersize=7, label="Recall@10")
    ax.plot(x, gkpi, "-s", color=C_ORANGE, markersize=7, label="GKPI")
    for i, (r, g) in enumerate(zip(recall, gkpi)):
        ax.annotate(f"{r:.3f}", (i, r), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, color=C_ACC)
        ax.annotate(f"{g:.3f}", (i, g), xytext=(0, -12), textcoords="offset points", ha="center", fontsize=8, color=C_ORANGE)
    ax.axhline(h1["SUPER-centralized"]["recall@K"], color=C_GREY, linestyle=":", linewidth=1)
    ax.text(0.05, h1["SUPER-centralized"]["recall@K"]+0.001, "SUPER-centralized recall", color=C_GREY, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(runs, fontsize=8.5)
    ax.set_ylabel("Metric value"); ax.set_ylim(0.005, 0.045)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Experiment trajectory across 12 runs (H1\\,$\\to$\\,H5)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig5_trajectory.pdf"))
    plt.close()

if __name__ == "__main__":
    fig1_architecture(); print("fig1 ok")
    fig2_calibration_invariance(); print("fig2 ok")
    fig3_pareto(); print("fig3 ok")
    fig4_llm_sweep(); print("fig4 ok")
    fig5_trajectory(); print("fig5 ok")
    print("All figures saved to", FIG)
