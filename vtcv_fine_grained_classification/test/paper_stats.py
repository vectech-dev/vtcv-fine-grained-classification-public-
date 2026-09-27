import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def load_run(exp_dir, col):
    path = Path(exp_dir) / "test_predictions" / "per_image.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found, run test mode for {exp_dir} first")
    df = pd.read_csv(path).sort_values("idx")
    return df["label"].to_numpy(), df[col].to_numpy()


def macro_scores(y, p, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm)
    n_pred, n_true = cm.sum(0), cm.sum(1)
    prec = np.divide(tp, n_pred, out=np.zeros(k), where=n_pred > 0)
    rec = np.divide(tp, n_true, out=np.zeros(k), where=n_true > 0)
    return 100 * prec.mean(), 100 * rec.mean()


def mcnemar(y, pa, pb):
    ok_a, ok_b = pa == y, pb == y
    b = int(np.sum(ok_a & ~ok_b))
    c = int(np.sum(~ok_a & ok_b))
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def holm(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(pvals)):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def summarize(name, dirs, col, k):
    preds = [load_run(d, col) for d in dirs]
    y0 = preds[0][0]
    for d, (y, _) in zip(dirs, preds):
        if not np.array_equal(y, y0):
            raise ValueError(f"test set order differs in {d}")
    scores = np.array([macro_scores(y, p, k) for y, p in preds])
    median = int(np.argsort(scores[:, 0])[len(dirs) // 2])
    sd = scores.std(0, ddof=1) if len(dirs) > 1 else np.zeros(2)
    return {
        "row": {"config": name, "runs": len(dirs),
                "mAP": scores[:, 0].mean(), "mAP_sd": sd[0],
                "mAR": scores[:, 1].mean(), "mAR_sd": sd[1],
                "median_run": dirs[median]},
        "y": y0,
        "p": preds[median][1],
        "mAP_mean": scores[:, 0].mean(),
    }


def compare(a, b, ra, rb, k, n_boot, rng):
    y = ra["y"]
    if not np.array_equal(y, rb["y"]):
        raise ValueError(f"test set order differs between {a} and {b}")
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = macro_scores(y[idx], ra["p"][idx], k)[0] - macro_scores(y[idx], rb["p"][idx], k)[0]
    point = macro_scores(y, ra["p"], k)[0] - macro_scores(y, rb["p"], k)[0]
    n_ab, n_ba, p = mcnemar(y, ra["p"], rb["p"])
    return {"comparison": f"{a} vs {b}",
            "dmAP_mean_runs": ra["mAP_mean"] - rb["mAP_mean"],
            "dmAP_median_run": point,
            "ci_low": np.percentile(diffs, 2.5), "ci_high": np.percentile(diffs, 97.5),
            "only_a_correct": n_ab, "only_b_correct": n_ba,
            "acc_diff_images": n_ab - n_ba,
            "p_raw": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default="paper_stats")
    args = ap.parse_args()

    cfg = json.loads(Path(args.runs).read_text())
    k = cfg.get("num_classes", 32)
    col = cfg.get("pred_column", "pred")
    n_boot = cfg.get("bootstrap", 10000)
    rng = np.random.default_rng(cfg.get("seed", 0))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = {name: summarize(name, dirs, col, k) for name, dirs in cfg["configs"].items()}
    table = pd.DataFrame([r["row"] for r in runs.values()]).round(2)
    table.to_csv(out / "summary.csv", index=False)
    print(table.to_string(index=False), "\n")

    for group, pairs in cfg["groups"].items():
        rows = [compare(a, b, runs[a], runs[b], k, n_boot, rng) for a, b in pairs]
        df = pd.DataFrame(rows)
        df["p_holm"] = holm(df["p_raw"])
        df["significant"] = df["p_holm"] < 0.05
        df = df.round(4)
        df.to_csv(out / f"{group}.csv", index=False)
        print(f"== {group} (Holm over {len(df)} comparisons) ==")
        print(df.to_string(index=False), "\n")


if __name__ == "__main__":
    main()
