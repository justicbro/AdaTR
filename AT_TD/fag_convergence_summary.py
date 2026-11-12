"""Summarize fag_convergence_check.py runs: Theorem 5.5 curves vs K and the Theorem 5.6 numbers at t*.

    python AT_TD/fag_convergence_summary.py AT_TD/fag_convergence_results/w256
"""
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BOUND_COLOR, MEASURED_COLOR = "#eb6834", "#2a78d6"  # categorical slots 2 and 1 (validated, light surface)
INK, MUTED, GRID = "#262624", "#6b6a63", "#e6e5e0"


def curves(d):
    eta, eps, delta, rho = float(d["eta"]), float(d["eps"]), d["delta"], d["rho"]
    K = np.arange(1, len(delta) + 1)
    bound = (1.0 / K + np.cumsum(rho) / K) / (2 * eta)
    measured = np.cumsum(delta ** 2) / K / (eta ** 2 * eps)
    return K, bound, measured


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.3g} ± {values.std():.2g}" if len(values) > 1 else f"{values[0]:.3g}"


def summarize(folder):
    runs = [dict(np.load(p, allow_pickle=True)) for p in sorted(glob.glob(os.path.join(folder, "*.npz")))]
    groups = {}
    for d in runs:
        groups.setdefault((str(d["kind"]), float(d["eta"])), []).append(d)

    for (kind, eta), ds in sorted(groups.items()):
        items = ", ".join(str(int(d["item"])) for d in ds)
        print(f"\n=== {kind}s [{items}], eta = {eta:g} (2*eta = {2 * eta:g}), shape {tuple(ds[0]['shape'])}, "
              f"eps = {mean_std([d['eps'] for d in ds])} ===")
        last = [curves(d) for d in ds]
        print(f"rho_bar_K at K=K_max        : {mean_std([d['rho'].mean() for d in ds])}   (2*eta = {2 * eta:g})")
        print(f"bound (1/K+rho_bar)/(2 eta)  : {mean_std([c[1][-1] for c in last])}")
        print(f"measured mean |D|^2/(eta^2 e): {mean_std([c[2][-1] for c in last])}")
        print(f"min over K of bound/measured : {mean_std([np.min(c[1] / c[2]) for c in last])}")
        print(f"min slack item 1 / (C.1) / eps: {min(d['item1_slack'].min() for d in ds):.2e} / "
              f"{min(d['c1_slack'].min() for d in ds):.2e}   (>= 0 means the inequality holds)")
        if "eps_g" in ds[0]:
            print(f"t*                           : {[int(d['tstar']) for d in ds]}")
            print(f"eps_g at t*                  : {mean_std([d['eps_g'] for d in ds])}")
            print(f"eps_g / |grad g|             : {mean_std([d['eps_g'] / d['grad_g_norm'] for d in ds])}")
            print(f"cos(grad g, grad g~)         : {mean_std([d['cos'] for d in ds])}")
            print(f"|G_eta(E^t*)| (autodiff)     : {mean_std([d['G_eta_norm'] for d in ds])}")
            thm = [np.sqrt(float(d['eps']) * (1 / len(d['delta']) + d['rho'].mean()) / (2 * eta)) + float(d['eps_g'])
                   for d in ds]
            print(f"Thm 5.6 bound at t*          : {mean_std(thm)}")

    plot(groups, folder)


def plot(groups, folder):
    keys = sorted(groups)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.2), squeeze=False, sharey=False)
    for ax, key in zip(axes[0], keys):
        kind, eta = key
        stack = [curves(d) for d in groups[key]]
        K = stack[0][0]
        for idx, label, color, style in ((1, r"bound $(1/K+\bar\rho_K)/(2\eta)$", BOUND_COLOR, "--"),
                                         (2, r"measured $\frac{1}{K}\sum_{t<K}\|\Delta_t\|_F^2/(\eta^2\epsilon)$",
                                          MEASURED_COLOR, "-")):
            Y = np.stack([c[idx] for c in stack])
            mean = Y.mean(0)
            ax.plot(K, mean, style, color=color, lw=2, label=label)
            if len(Y) > 1:
                ax.fill_between(K, Y.min(0), Y.max(0), color=color, alpha=0.15, lw=0)
        ax.axhline(1.0, color=MUTED, lw=1, ls=":")
        ax.text(K[-1], 0.9, "trivial level 1", color=MUTED, fontsize=8, ha="right", va="top")
        ax.set_yscale("log")
        ax.set_xlabel("K", color=INK)
        n = len(groups[key])
        ax.set_title(f"{kind}s (n={n}), $\\eta$={eta:g}", color=INK, fontsize=10)
        ax.grid(True, which="major", color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(GRID)
    axes[0][0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(folder, f"fag_theorem55_curves.{ext}"), dpi=200, bbox_inches="tight")
    print(f"\nfigure: {os.path.join(folder, 'fag_theorem55_curves.pdf')}")


if __name__ == "__main__":
    summarize(sys.argv[1])
