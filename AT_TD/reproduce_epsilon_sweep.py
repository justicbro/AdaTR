"""Re-run the paper's epsilon sweep (epsilon_vs_loss_mean_rank_{5,10}) with the original epsilon_comp3.py code.

epsilon_comp3.py is loaded unchanged except for its hard-coded CUDA_VISIBLE_DEVICES="7" and its __main__ block,
so AdaTR, FAG-AdaTR and ATTR are exactly the versions behind epgilon_compare_results11 (rank 5) and 14 (rank 10).
One job runs one rank on a subset of images; --aggregate merges the jobs and compares them with the saved results.

Examples (from the project root):
    CUDA_VISIBLE_DEVICES=0 python AT_TD/reproduce_epsilon_sweep.py --rank 5 --images 1.png --out AT_TD/reproduce_epsilon_sweep_results/rank5_1.npz
    python AT_TD/reproduce_epsilon_sweep.py --aggregate AT_TD/reproduce_epsilon_sweep_results
"""
import argparse
import glob
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = {5: "epgilon_compare_results11/results_data.npz", 10: "epgilon_compare_results14/results_data.npz"}
METHODS = [("ours", "AdaTR"), ("oursF", "FAG-AdaTR"), ("ATR", "ATTR")]


def load_epsilon_comp3():
    path = os.path.join(HERE, "epsilon_comp3.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    gpu_line = 'os.environ["CUDA_VISIBLE_DEVICES"] = "7"'
    assert src.count(gpu_line) == 1, "epsilon_comp3.py changed; update the runner"
    src = src.replace(gpu_line, "pass  # GPU chosen by the caller")
    src = src.split('\nif __name__ == "__main__":')[0]
    sys.path.insert(0, HERE)
    module = types.ModuleType("epsilon_comp3")
    module.__file__ = path
    exec(compile(src, path, "exec"), module.__dict__)
    return module


def run(args):
    import torch

    ec = load_epsilon_comp3()
    # The paper runs shared one random set of TR cores across all images and epsilons; fix it with a seed here.
    torch.manual_seed(args.seed)
    cores = [c.to("cuda") for c in ec.tr.initialize_tr_cores([128, 128, 3], [args.rank] * 4, device="cpu")]
    config = ec.OptimizationConfig(epsilon=args.epsilons[0], lr=0.05, outer_num_steps=100, inner_num_steps=100,
                                   outer_dis_num=1, inner_dis_num=2000, inner_tol=1e-10, tr_cores=cores)
    finals = {key: np.zeros((len(args.images), len(args.epsilons))) for key, _ in METHODS}
    curves = {key: np.zeros((len(args.images), len(args.epsilons), 100)) for key, _ in METHODS}
    trd = np.zeros(len(args.images))
    out_dir = os.path.dirname(args.out) or "."
    for i, name in enumerate(args.images):
        for j, eps in enumerate(args.epsilons):
            config.epsilon = float(eps)
            torch.manual_seed(args.seed + 1000 * i + j)
            _, ours, oursF, atr, trd_val, c_ours, c_oursF, c_atr = ec.process_image(
                os.path.join(args.image_dir, name), out_dir, config)
            for key, last, curve in (("ours", ours, c_ours), ("oursF", oursF, c_oursF), ("ATR", atr, c_atr)):
                finals[key][i, j] = last
                curves[key][i, j, :len(curve)] = curve
            trd[i] = trd_val
            print(f"[reproduce] rank={args.rank} {name} eps={eps:g}: AdaTR={ours:.4f} FAG-AdaTR={oursF:.4f} "
                  f"ATTR={atr:.4f} TRD={trd_val:.4f}", flush=True)
    np.savez(args.out, rank=args.rank, images=np.array(args.images), epsilons=np.array(args.epsilons, dtype=float),
             TRD=trd, **{f"{k}_final": v for k, v in finals.items()}, **{f"{k}_curves": v for k, v in curves.items()})


def aggregate(folder):
    by_rank = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        d = np.load(path, allow_pickle=True)
        by_rank.setdefault(int(d["rank"]), []).append(d)
    for rank, parts in sorted(by_rank.items()):
        images = np.concatenate([p["images"] for p in parts])
        eps = parts[0]["epsilons"]
        new = {key: np.concatenate([p[f"{key}_final"] for p in parts]) for key, _ in METHODS}
        new_trd = np.concatenate([p["TRD"] for p in parts])
        os.makedirs(os.path.join(folder, "merged"), exist_ok=True)
        np.savez(os.path.join(folder, "merged", f"rank{rank}.npz"), images=images, epsilons=eps, TRD=new_trd, **new)
        ref = np.load(os.path.join(HERE, REFERENCE[rank]), allow_pickle=True)
        # The paper run used the 8 images in sorted order 1.png..8.png; compare the same images only.
        idx = [int(os.path.splitext(name)[0]) - 1 for name in images]
        old = {key: ref[f"{key}_final_all"][idx] for key, _ in METHODS}
        old_trd = ref["TRD_all"][idx]
        print(f"\n=== rank {rank}: images {', '.join(images)}, mean over these images (paper run -> rerun) ===")
        print(f"TRD (no attack): {old_trd.mean():.3f} -> {new_trd.mean():.3f}")
        print("eps    " + "".join(f"{label:>22s}" for _, label in METHODS))
        for j, e in enumerate(eps):
            row = "".join(f"{old[key][:, j].mean():9.3f} -> {new[key][:, j].mean():8.3f}  " for key, _ in METHODS)
            print(f"{e:5g}  {row}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", metavar="FOLDER")
    parser.add_argument("--image-dir", default="./AT_TD/My picture")
    parser.add_argument("--images", nargs="+", default=[f"{i}.png" for i in range(1, 9)])
    parser.add_argument("--rank", type=int, choices=sorted(REFERENCE), default=5)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[5.0 * i for i in range(1, 10)])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.aggregate:
        aggregate(args.aggregate)
    else:
        if not args.out:
            parser.error("--out is required unless --aggregate is given")
        run(args)


if __name__ == "__main__":
    main()
