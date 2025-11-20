#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, math, random
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt

# ---------- IO / selection helpers ----------

def load_traces(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        rows = json.load(f)
    # normalize a few fields we rely on
    for r in rows:
        # Handle different JSON structures
        if isinstance(rows, dict) and "signals" in rows:
            rows = rows["signals"]
            break
    
    if not isinstance(rows, list):
        rows = [rows] if isinstance(rows, dict) else []
        
    for r in rows:
        # predicted index / vector
        if "ensemble_final_prob" in r:
            prob_data = r["ensemble_final_prob"]
            if isinstance(prob_data, list):
                if len(prob_data) > 0 and isinstance(prob_data[0], list):
                    # Handle nested list structure
                    r["_final_prob_vec"] = np.array(prob_data[0])
                else:
                    # Handle flat list structure
                    r["_final_prob_vec"] = np.array(prob_data)
            else:
                r["_final_prob_vec"] = np.array(prob_data)
        else:
            # fallback: mean of per_model_probs
            if "per_model_probs" in r and len(r["per_model_probs"]) > 0:
                vecs = np.array(list(r["per_model_probs"].values()))
                r["_final_prob_vec"] = vecs.mean(axis=0)
            else:
                r["_final_prob_vec"] = None

        # predicted class idx/name
        if "pred_idx" in r:
            r["_pred_idx"] = int(r["pred_idx"])
        elif r.get("_final_prob_vec") is not None:
            r["_pred_idx"] = int(np.argmax(r["_final_prob_vec"]))
        else:
            r["_pred_idx"] = None

        # true idx if present
        if "true_idx" in r and r["true_idx"] is not None:
            r["_true_idx"] = int(r["true_idx"])
        else:
            r["_true_idx"] = None

        # correctness
        if "correct" in r:
            r["_correct"] = bool(r["correct"])
        elif r["_true_idx"] is not None and r["_pred_idx"] is not None:
            r["_correct"] = (r["_true_idx"] == r["_pred_idx"])
        else:
            r["_correct"] = None

        # snr for labeling
        r["_snr"] = r.get("snr_db", r.get("snr", None))
    return rows


def choose_examples(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Pick one correct and one incorrect for vote timelines,
    and choose 3 'hardest' cases for Shapley bar plots.

    Hardness heuristic:
      - if incorrect: higher ensemble confidence in the wrong class = harder
      - if correct: lower ensemble confidence in the right class = harder
    """
    corrects = [r for r in rows if r.get("_correct") is True and r.get("_final_prob_vec") is not None and r.get("_pred_idx") is not None]
    incorrects = [r for r in rows if r.get("_correct") is False and r.get("_final_prob_vec") is not None and r.get("_pred_idx") is not None]

    # choose representative correct/incorrect
    correct = None
    if corrects:
        correct = min(corrects, key=lambda r: float(r["_final_prob_vec"][r["_pred_idx"]]))  # lowest conf correct → illustrative
    wrong = None
    if incorrects:
        wrong = max(incorrects, key=lambda r: float(r["_final_prob_vec"][r["_pred_idx"]]))  # highest conf wrong → illustrative

    # difficulty score for ranking (both groups)
    def diff_score(r):
        vec = r["_final_prob_vec"]
        if vec is None or r["_pred_idx"] is None:
            return -1e9
        if r["_correct"] is False:
            return float(vec[r["_pred_idx"]])                 # high-conf wrong = hard
        elif r["_correct"] is True and r["_true_idx"] is not None:
            return float(1.0 - vec[r["_true_idx"]])           # low-conf right = hard
        else:
            # fallback: low max prob = hard
            return float(1.0 - float(np.max(vec)))

    ranked = sorted([r for r in rows if r.get("_final_prob_vec") is not None and r.get("_pred_idx") is not None],
                    key=diff_score, reverse=True)
    hardest = ranked[:3]
    return correct, wrong, hardest


# ---------- plotting ----------

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def plot_vote_timeline(sample: Dict[str, Any], figpath: Path, top_k: int = 10, title_prefix: str = "Vote Timeline"):
    pmp: Dict[str, List[float]] = sample.get("per_model_probs", {})
    contrib: Dict[str, float] = sample.get("shapley_contribution", {})
    if not pmp:
        raise ValueError("Sample missing per_model_probs")
    if not contrib:
        # still allow plotting as a plain vote bar if Shapley missing
        contrib = {k: 0.0 for k in pmp.keys()}

    # sort models by contribution desc
    ordered = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in ordered][:min(top_k, len(ordered))]

    # use predicted index for confidence bars
    pred_idx = sample.get("_pred_idx", None)
    if pred_idx is None:
        raise ValueError("Sample missing predicted index/_pred_idx")

    confs = []
    colors = []
    for name in names:
        vec = pmp.get(name)
        if vec is None:
            # allow base name without suffix
            base = name.split("_m")[0]
            vec = pmp.get(base)
        if vec is None:
            confs.append(0.0)
            colors.append("gray")
            continue
        confs.append(float(vec[pred_idx]))
        colors.append("green" if contrib.get(name, 0.0) >= 0 else "red")

    final_p = None
    if sample.get("_final_prob_vec") is not None:
        final_p = float(sample["_final_prob_vec"][pred_idx])

    plt.figure(figsize=(10, 6))
    xs = np.arange(len(names))
    bars = plt.bar(xs, confs, color=colors, edgecolor='black', alpha=0.85)
    if final_p is not None:
        plt.axhline(final_p, linestyle="--", linewidth=3, label=f"Ensemble p* = {final_p:.3f}")

    tc = sample.get("_true_idx", None)
    hdr = f"{title_prefix} • Pred idx={pred_idx}"
    if tc is not None:
        hdr += f" • True idx={tc} • {'Correct' if pred_idx==tc else 'Misclassified'}"
    if sample.get('_snr') is not None:
        hdr += f" • SNR={sample['_snr']} dB"
    plt.title(hdr, pad=16)
    plt.ylabel("Confidence in predicted class")
    plt.ylim(0, 1.05)
    plt.xticks(xs, names, rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    if final_p is not None:
        plt.legend(loc="lower right")
    # value labels
    for bx, v in zip(bars, confs):
        plt.text(bx.get_x() + bx.get_width()/2, bx.get_height()+0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(figpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_shapley_bar(sample: Dict[str, Any], figpath: Path, top_k: int = 12, title: str = "Model contributions (Shapley)"):
    contrib: Dict[str, float] = sample.get("shapley_contribution", {})
    if not contrib:
        raise ValueError("Sample missing shapley_contribution for bar plot")
    ordered = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    labels = [k for k, _ in ordered][::-1]   # small→large on y
    vals = [float(v) for _, v in ordered][::-1]
    colors = ["green" if v >= 0 else "red" for v in vals]

    plt.figure(figsize=(9, 6))
    y = np.arange(len(labels))
    plt.barh(y, vals, color=colors, edgecolor="black", alpha=0.9)
    plt.yticks(y, labels)
    plt.xlabel("Δp contribution (normalized)")
    hdr = title
    tc = sample.get("_true_idx", None); pc = sample.get("_pred_idx", None)
    if pc is not None:
        hdr += f" • pred={pc}"
    if tc is not None:
        hdr += f" • true={tc}"
    if sample.get("_snr") is not None:
        hdr += f" • SNR={sample['_snr']} dB"
    plt.title(hdr, pad=14)
    plt.grid(True, axis='x', alpha=0.25)
    for yi, v in zip(y, vals):
        plt.text(v + (0.01 if v>=0 else -0.01), yi, f"{v:+.3f}", ha="left" if v>=0 else "right",
                 va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(figpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_disagreement_heatmap(rows: List[Dict[str, Any]], figpath: Path, sample_n: int = 50000, seed: int = 1337):
    rng = random.Random(seed)
    pool = [r for r in rows if isinstance(r.get("per_model_probs"), dict) and len(r["per_model_probs"]) >= 2 and r.get("_pred_idx") is not None]
    if not pool:
        raise ValueError("No rows with per_model_probs + _pred_idx available for heatmap.")
    if len(pool) > sample_n:
        pool = rng.sample(pool, sample_n)

    # discover a stable model list (preserve insertion order from first row)
    canonical_names = list(pool[0]["per_model_probs"].keys())
    k = len(canonical_names)
    mat = np.zeros((k, k), dtype=float)
    count = 0

    for r in pool:
        pmp: Dict[str, List[float]] = r["per_model_probs"]
        pred_idx = r["_pred_idx"]
        # build vector aligned to canonical_names; skip row if any missing
        if any(name not in pmp for name in canonical_names):
            continue
        probs = np.array([pmp[name][pred_idx] for name in canonical_names], dtype=float)
        diff = np.abs(probs[:, None] - probs[None, :])
        mat += diff
        count += 1

    if count == 0:
        raise ValueError("All candidate rows missing one or more canonical model names.")
    mat /= float(count)

    plt.figure(figsize=(8.8, 7.2))
    im = plt.imshow(mat, cmap="viridis", interpolation="nearest")
    plt.title(f"Pairwise Disagreement Heatmap (mean |p_i(c*)-p_j(c*)|)\n(n={count:,} samples)", pad=16)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(np.arange(k), canonical_names, rotation=45, ha='right')
    plt.yticks(np.arange(k), canonical_names)
    # gridlines
    for i in range(k):
        plt.axhline(i-0.5, color="white", linewidth=0.5, alpha=0.4)
        plt.axvline(i-0.5, color="white", linewidth=0.5, alpha=0.4)
    # annotate
    for i in range(k):
        for j in range(k):
            plt.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", color="white", fontsize=8, weight="bold")
    plt.tight_layout()
    plt.savefig(figpath, dpi=300, bbox_inches="tight")
    plt.close()


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/vote_traces.json")
    ap.add_argument("--figdir", default="figs")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--heatmap-n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rows = load_traces(args.in_path)
    figdir = Path(args.figdir); _ensure_dir(figdir)

    # choose samples
    correct, wrong, hardest = choose_examples(rows)

    # vote timelines
    if correct is not None:
        plot_vote_timeline(correct, figdir / "vote_timeline_correct.pdf", top_k=args.topk, title_prefix="Vote Timeline (Correct)")
    if wrong is not None:
        plot_vote_timeline(wrong, figdir / "vote_timeline_incorrect.pdf", top_k=args.topk, title_prefix="Vote Timeline (Incorrect)")

    # shapley bars for 3 hardest cases
    for i, sample in enumerate(hardest, 1):
        plot_shapley_bar(sample, figdir / f"shapley_bar_hardcase_{i}.pdf", top_k=max(8, args.topk))

    # disagreement heatmap
    plot_disagreement_heatmap(rows, figdir / "disagreement_heatmap.pdf", sample_n=args.heatmap_n, seed=args.seed)

    print(f"[xai-figs] Wrote figures to: {figdir.resolve()}")

if __name__ == "__main__":
    main()