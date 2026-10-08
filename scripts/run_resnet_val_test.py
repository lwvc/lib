# run_resnet_val_test.py
# Only ResNet, inputs:
#   C:\Lab\model_val_test_train\ResNet\test\test.csv + *.nii
#   C:\Lab\model_val_test_train\ResNet\val\val.csv + *.nii

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from src.model_registry import get_model_config
from src.models.resnet import ResBlock, NetResBlock, RESNET_MODEL
from src.predict import predict_batch


def _alias_resnet_for_unpickle():
    # Checkpoints were saved with torch.save(full model) where classes
    # lived in __main__ (training script). Unpickling therefore looks up
    # __main__.RESNET_MODEL / NetResBlock / ResBlock. Alias them to the
    # current implementations so torch.load works regardless of entry script.
    import __main__ as _main

    for _name, _cls in (
        ("ResBlock", ResBlock),
        ("NetResBlock", NetResBlock),
        ("RESNET_MODEL", RESNET_MODEL),
    ):
        if not hasattr(_main, _name):
            setattr(_main, _name, _cls)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESNET_ROOT = Path(r"C:\Lab\model_val_test_train\ResNet")

SPLITS = {
    "test": {
        "csv": RESNET_ROOT / "test" / "test.csv",
        "image_dir": RESNET_ROOT / "test",
    },
    "val": {
        "csv": RESNET_ROOT / "val" / "val.csv",
        "image_dir": RESNET_ROOT / "val",
    },
}

RESNET_MODELS = [
    "resnet_t1",
    "resnet_t1_sex",
    "resnet_t1_b0",
    "resnet_t1_sex_b0",
]

OUTPUT_PATH = Path(r"C:\Lab\brain_age_test_retest\output\ResNet_val_test.xlsx")

VARIANT_MAP = {
    "resnet_t1": "T1",
    "resnet_t1_sex": "T1_SEX",
    "resnet_t1_b0": "T1_B0",
    "resnet_t1_sex_b0": "T1_SEX_B0",
}


class ResNetValTestDataset(Dataset):
    """Dataset driven by test.csv / val.csv + image folder."""

    def __init__(self, csv_path, image_dir):
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.df = pd.read_csv(csv_path)
        # Normalize column names (strip quotes/spaces just in case)
        self.df.columns = [c.strip() for c in self.df.columns]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = str(row["Filename"]).strip()
        image_path = self.image_dir / filename
        if not image_path.exists():
            raise FileNotFoundError(f"找不到影像:\n{image_path}")

        image = nib.load(str(image_path)).get_fdata().astype(np.float32)
        # Handle (193,229,193,1) trailing singleton -> (193,229,193)
        image = np.squeeze(image)
        image = torch.from_numpy(image).unsqueeze(0)

        # CSV already uses SEX(0=Male,1=Female)
        sex_val = float(row["SEX(0=Male,1=Female)"])
        # Scanner: 1.5 or 3 -> int like original code
        scanner_val = int(float(row["Scanner"]))
        age = float(row["Age"])

        return {
            "image": image,
            "age": torch.tensor(age, dtype=torch.float32),
            "subject_id": str(row["Subject_ID"]),
            "filename": filename,
            "sex": torch.tensor([sex_val], dtype=torch.float32),
            "scanner": torch.tensor([scanner_val], dtype=torch.float32),
        }


class TestBalancedAgeBatchSampler(Sampler):
    """Copy of original test sampler, batch_size=1."""

    def __init__(self, ages, batch_size=1):
        self.batch_size = batch_size
        ages = np.asarray(ages, dtype=float)
        quantiles = np.linspace(0, 1, batch_size + 1)
        bins = np.quantile(ages, quantiles)
        self.age_bins = [(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
        self.age_bins[-1] = (self.age_bins[-1][0], self.age_bins[-1][1] + 0.1)
        groups = {b: [] for b in self.age_bins}
        for idx, age in enumerate(ages):
            for b in self.age_bins:
                if b[0] <= age < b[1]:
                    groups[b].append(idx)
                    break
        batches = []
        while any(groups.values()):
            batch = []
            for b in self.age_bins:
                if groups[b]:
                    batch.append(groups[b].pop(0))
                if len(batch) == self.batch_size:
                    break
            if batch:
                batches.append(batch)
        self.batches = batches

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)


def build_model(model_config):
    _alias_resnet_for_unpickle()
    model_type = model_config["model_type"]
    if model_type == "cat":
        model = RESNET_MODEL(model_type=model_type, cat_dim=model_config["cat_dim"])
    else:
        model = RESNET_MODEL(model_type=model_type)
    model = model.to(DEVICE)
    pretrained = torch.load(model_config["checkpoint"], map_location=DEVICE, weights_only=False)
    state_dict = pretrained.state_dict() if hasattr(pretrained, "state_dict") else pretrained
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def run_one(split_name, model_name):
    config = get_model_config(model_name)
    split = SPLITS[split_name]
    print("=" * 70)
    print(f"Split : {split_name} ({split['csv']})")
    print(f"Model : {model_name}")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    if not config["checkpoint"].exists():
        raise FileNotFoundError(f"找不到模型權重:\n{config['checkpoint']}")

    model = build_model(config)
    dataset = ResNetValTestDataset(split["csv"], split["image_dir"])
    ages = dataset.df["Age"].to_numpy(dtype=float)
    sampler = TestBalancedAgeBatchSampler(ages, batch_size=1)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

    subject_ids, true_ages, pred_ages, filenames = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            predictions = predict_batch(model=model, batch=batch, model_config=config, device=DEVICE)
            if torch.is_tensor(predictions):
                predictions = predictions.detach().cpu().numpy().reshape(-1).tolist()
            else:
                predictions = np.asarray(predictions).reshape(-1).tolist()

            batch_subject_ids = [str(x) for x in batch["subject_id"]]
            batch_filenames = [str(x) for x in batch["filename"]]
            batch_true = batch["age"].detach().cpu().numpy().reshape(-1).tolist()

            subject_ids.extend(batch_subject_ids)
            filenames.extend(batch_filenames)
            true_ages.extend(batch_true)
            pred_ages.extend(predictions)

    result_df = pd.DataFrame({
        "Subject ID": subject_ids,
        "True Age": true_ages,
        "Predicted Age": pred_ages,
        "File Name": filenames,
    })
    sheet_name = f"{split_name}_{VARIANT_MAP[model_name]}"[:31]
    return sheet_name, result_df


def main():
    print(f"目前使用裝置: {DEVICE}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    results = []
    for split_name in ["test", "val"]:
        for model_name in RESNET_MODELS:
            sheet_name, result_df = run_one(split_name, model_name)
            results.append((sheet_name, result_df))
            print(f"完成: {sheet_name}, 樣本數: {len(result_df)}")

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl", mode="w") as writer:
        for sheet_name, result_df in results:
            result_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\n全部完成，輸出: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
