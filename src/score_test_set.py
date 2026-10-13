from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save images without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

CKPT = "models/patchcore_metal_nut.ckpt"
HEAT_DIR = Path("outputs/heatmaps")


def to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def display_image(img_tensor):
    """Turn the model's input tensor back into a viewable RGB image."""
    img = to_np(img_tensor).transpose(1, 2, 0)          # C,H,W -> H,W,C
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def save_heatmap(img, amap, out_path, score, defect_type):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))
    axes[0].imshow(img)
    axes[0].set_title(f"actual: {defect_type}", fontsize=11)
    axes[1].imshow(img)
    axes[1].imshow(amap, cmap="inferno", alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title(f"anomaly score: {score:.2f}", fontsize=11)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)


def main():
    HEAT_DIR.mkdir(parents=True, exist_ok=True)

    datamodule = MVTecAD(
        root="data/MVTecAD", category="metal_nut",
        eval_batch_size=32, num_workers=0,
    )
    engine = Engine(accelerator="cpu", devices=1, default_root_dir="models/runs")

    print("Scoring test images with the saved model...")
    predictions = engine.predict(model=Patchcore(), datamodule=datamodule, ckpt_path=CKPT)

    rows = []
    for batch in predictions:
        scores = to_np(batch.pred_score).reshape(-1)
        maps = to_np(batch.anomaly_map)
        for i, path in enumerate(batch.image_path):
            path = Path(path)
            defect_type = path.parent.name
            part_id = f"{defect_type}_{path.stem}"
            heat_path = HEAT_DIR / f"{part_id}.png"

            save_heatmap(display_image(batch.image[i]), maps[i].squeeze(),
                         heat_path, float(scores[i]), defect_type)

            rows.append({
                "part_id": part_id,
                "image_path": str(path),
                "defect_type": defect_type,
                "is_defect": int(defect_type != "good"),
                "score": float(scores[i]),
                "heatmap_path": str(heat_path),
            })

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    df.to_csv("outputs/scores.csv", index=False)
    print(f"\nSaved {len(df)} rows to outputs/scores.csv")
    print(f"Saved {len(df)} heatmaps to {HEAT_DIR}/")

    # --- Summary ---
    good = df[df.is_defect == 0].score
    bad = df[df.is_defect == 1].score
    print("\nSCORE SUMMARY")
    print(f"  Good parts:      min {good.min():.2f}   max {good.max():.2f}   mean {good.mean():.2f}")
    print(f"  Defective parts: min {bad.min():.2f}   max {bad.max():.2f}   mean {bad.mean():.2f}")
    print("\n  Mean score by type:")
    print(df.groupby("defect_type").score.mean().sort_values().round(2).to_string())

    missed = df[(df.is_defect == 1) & (df.score < 0.5)]
    false_alarms = df[(df.is_defect == 0) & (df.score >= 0.5)]
    print(f"\n  At the default 0.5 threshold:")
    print(f"    defects missed:  {len(missed)}")
    print(f"    false alarms:    {len(false_alarms)}")

    print("\n  Hardest cases (closest to the boundary):")
    print("    lowest-scoring defect: ", df[df.is_defect == 1].nsmallest(1, "score")[["part_id", "score"]].values[0])
    print("    highest-scoring good:  ", df[df.is_defect == 0].nlargest(1, "score")[["part_id", "score"]].values[0])

    # --- Score distribution chart ---
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(good, bins=bins, color="#2F6FDB", alpha=0.85, label=f"good ({len(good)})")
    ax.hist(bad, bins=bins, color="#F28C28", alpha=0.85, label=f"defective ({len(bad)})")
    ax.axvline(0.5, color="#333333", linestyle="--", linewidth=1)
    ax.text(0.51, ax.get_ylim()[1] * 0.9, "default threshold", fontsize=9)
    ax.set_xlabel("anomaly score")
    ax.set_ylabel("number of parts")
    ax.set_title("How well the score separates good and defective parts")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig("outputs/score_distribution.png", dpi=120)
    print("\nSaved outputs/score_distribution.png")


if __name__ == "__main__":
    main()
