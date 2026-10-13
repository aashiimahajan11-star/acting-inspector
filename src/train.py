import json
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore


def main():
    # Data: uses the files already in data/MVTecAD (no new download)
    datamodule = MVTecAD(
        root="data/MVTecAD",
        category="metal_nut",
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=0,          # safest setting on a Mac
    )

    # Model: PatchCore with its standard settings
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=("layer2", "layer3"),
        coreset_sampling_ratio=0.1,   # keep 10% of normal patches as memory
    )

    # Engine: runs on the CPU, which is reliable for PatchCore on a Mac
    engine = Engine(accelerator="cpu", devices=1, default_root_dir="models/runs")

    print("\n=== Training on good parts only ===")
    engine.fit(model=model, datamodule=datamodule)

    ckpt = Path("models/patchcore_metal_nut.ckpt")
    engine.trainer.save_checkpoint(ckpt)
    print(f"\nSaved model to {ckpt}")

    print("\n=== Evaluating on the test set ===")
    results = engine.test(model=model, datamodule=datamodule)

    metrics = {k: float(v) for k, v in results[0].items()}
    print("\nRESULTS")
    for k, v in metrics.items():
        print(f"  {k:20s} {v:.4f}")

    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nSaved metrics to outputs/test_metrics.json")


if __name__ == "__main__":
    main()
