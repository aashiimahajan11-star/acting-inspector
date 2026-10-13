from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path("data/MVTecAD/metal_nut")

def count(folder):
    return len(list(folder.glob("*.png")))

# --- 1. Counts ---
print("TRAIN")
for d in sorted((ROOT / "train").iterdir()):
    if d.is_dir():
        print(f"  {d.name:10s} {count(d)}")

print("TEST")
good = bad = 0
for d in sorted((ROOT / "test").iterdir()):
    if d.is_dir():
        n = count(d)
        print(f"  {d.name:10s} {n}")
        if d.name == "good":
            good += n
        else:
            bad += n
print(f"\nTest set: {good} good, {bad} defective ({bad / (good + bad):.0%} defect rate)")

# --- 2. Visual overview: one example per type, with the defect area highlighted ---
types = [d.name for d in sorted((ROOT / "test").iterdir()) if d.is_dir()]
fig, axes = plt.subplots(2, len(types), figsize=(3 * len(types), 6.5))

for i, t in enumerate(types):
    img_path = sorted((ROOT / "test" / t).glob("*.png"))[0]
    img = np.array(Image.open(img_path).convert("RGB"))

    axes[0, i].imshow(img)
    axes[0, i].set_title(t, fontsize=13, fontweight="bold")
    axes[0, i].axis("off")

    ax = axes[1, i]
    ax.imshow(img)
    ax.axis("off")
    if t == "good":
        ax.set_title("no defect", fontsize=10)
    else:
        mask_path = ROOT / "ground_truth" / t / f"{img_path.stem}_mask.png"
        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert("L")) > 0
            overlay = np.zeros((*mask.shape, 4))
            overlay[mask] = [1.0, 0.35, 0.0, 0.6]   # orange highlight
            ax.imshow(overlay)
        ax.set_title("defect area", fontsize=10)

plt.tight_layout()
out = Path("outputs/data_overview.png")
fig.savefig(out, dpi=120)
print(f"Saved {out}")
