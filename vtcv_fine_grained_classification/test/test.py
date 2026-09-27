import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vtcv_fine_grained_classification.src.configs.config import ExperimentationConfig
from vtcv_fine_grained_classification.src.configs.train_config import get_dataset
from vtcv_fine_grained_classification.src.models.build_model import build_model
from vtcv_fine_grained_classification.src.models.process_model import MainModel
from vtcv_fine_grained_classification.test.metrics import calculate_metrics_with_confidence, normalize_confusion_matrix
from vtcv_fine_grained_classification.test.test_utils import create_pdf_report
from vtcv_fine_grained_classification.train.train_utils import save_visualizations


def test(experiment_dir: str):
    exp_dir = Path(experiment_dir)
    config = ExperimentationConfig.parse_obj(json.loads((exp_dir / "TrainConfig.json").read_text()))
    device = config.device

    model = MainModel(build_model(config, device, load_tax=False), config).to(device)
    model.load_state_dict(torch.load(exp_dir / "model_best.pth", map_location=device))
    model.eval()

    dataset = get_dataset(config, "Test")
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False,
                        num_workers=config.num_workers, pin_memory=True)

    out_dir = exp_dir / "test_predictions"
    vis_dir = out_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_pred = [], []
    for b, (images, labels) in enumerate(tqdm(loader)):
        images, labels = images.to(device), labels.to(device)
        visualize = b < config.num_vis_batches_test
        _, out = model(images, labels, train=False, epoch=config.epochs, visualize=visualize)
        if visualize:
            save_visualizations(images, labels, out, b, str(vis_dir))
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(out["logits"].argmax(1).cpu().tolist())

    pd.DataFrame({"idx": range(len(y_true)), "label": y_true, "pred": y_pred}).to_csv(
        out_dir / "per_image.csv", index=False)

    class_names = [str(i) for i in range(config.num_classes)]
    metrics_df, macro, micro = calculate_metrics_with_confidence(y_true, y_pred, class_names)
    metrics_df.to_csv(out_dir / "per_class_metrics.csv", index=False)
    normalize_confusion_matrix(y_true, y_pred, exp_dir, norm="true")
    create_pdf_report(metrics_df, macro, micro, str(out_dir / "classification_metrics_report.pdf"))

    summary = {"mAP": 100 * macro["precision"], "mAR": 100 * macro["recall"], "n_test": len(y_true)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{config.exp_name}: mAP {summary['mAP']:.2f}  mAR {summary['mAR']:.2f}  ({len(y_true)} test images)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", required=True)
    args = parser.parse_args()
    test(args.exp_dir)
