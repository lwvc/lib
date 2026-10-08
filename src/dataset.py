# dataset.py

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


DATA_ROOT = Path(r"C:\Lab\test_retest_dataset")

DATASET_FILES = {
    "Maclaren": DATA_ROOT / "Maclaren" / "Maclaren.xlsx",
    "MMRR": DATA_ROOT / "MMRR" / "MMRR.xlsx",
    "OAS1": DATA_ROOT / "OAS1" / "OAS1.xlsx",
}

MODALITY_CONFIG = {
    "T1WI": {
        "folder": Path("ndf") / "rmi",
        "filename_col": "rmi",
    },
    "GMP": {
        "folder": Path("ndf") / "rp1",
        "filename_col": "rp1",
    },
    "WMP": {
        "folder": Path("ndf") / "rp2",
        "filename_col": "rp2",
    },

    # Deep grey matter
    # DGM requires additional image processing.
}


class BrainAgeDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        modality,
        crop_coords=None,
        return_metadata=False,
    ):
        if dataset_name not in DATASET_FILES:
            raise ValueError(
                f"未知 dataset: {dataset_name}. "
                f"可用: {list(DATASET_FILES)}"
            )

        if modality not in MODALITY_CONFIG:
            raise ValueError(
                f"未知 modality: {modality}. "
                f"可用: {list(MODALITY_CONFIG)}"
            )

        self.dataset_name = dataset_name
        self.modality = modality
        self.crop_coords = crop_coords
        self.return_metadata = return_metadata

        self.excel_path = DATASET_FILES[dataset_name]
        self.dataset_root = self.excel_path.parent

        config = MODALITY_CONFIG[modality]
        self.image_dir = self.dataset_root / config["folder"]
        self.filename_col = config["filename_col"]

        self.df = pd.read_excel(self.excel_path)

        self._validate()

    def _validate(self):
        required_cols = [
            "Subject_ID",
            "Age",
            "Sex (1=Male, 2=Female)",
            "Scanner",
            self.filename_col,
        ]

        missing = [
            col for col in required_cols
            if col not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"{self.excel_path.name} 缺少欄位: {missing}"
            )

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"影像資料夾不存在: {self.image_dir}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        filename = str(row[self.filename_col]).strip()

        if not filename.endswith(".nii"):
            filename += ".nii"

        image_path = self.image_dir / filename

        if not image_path.exists():
            raise FileNotFoundError(
                f"找不到影像:\n{image_path}"
            )

        image = nib.load(image_path).get_fdata().astype(np.float32)

        # --------------------------------------------
        # Crop
        # --------------------------------------------
        if self.crop_coords is not None:
            min_coords, max_coords = self.crop_coords

            image = image[
                min_coords[0]:max_coords[0] + 1,
                min_coords[1]:max_coords[1] + 1,
                min_coords[2]:max_coords[2] + 1,
            ]

        image = torch.from_numpy(image).unsqueeze(0)

        age = float(row["Age"])

        if not self.return_metadata:
            return image, age

        # ResNet / output analysis 可使用
        sex_raw = row["Sex (1=Male, 2=Female)"]
        sex = 0.0 if sex_raw == 1 else 1.0

        scanner = int(float(row["Scanner"]))

        return {
            "image": image,
            "age": age,
            "subject_id": str(row["Subject_ID"]),
            "filename": filename,
            "sex": torch.tensor([sex], dtype=torch.float32),
            "scanner": torch.tensor(
                [scanner],
                dtype=torch.float32,
            ),
        }


def get_dataset_info(dataset_name, modality):
    """
    方便其他程式取得 Excel / MRI folder。
    """
    excel_path = DATASET_FILES[dataset_name]

    config = MODALITY_CONFIG[modality]

    return {
        "excel_path": excel_path,
        "image_dir": excel_path.parent / config["folder"],
        "filename_col": config["filename_col"],
    }
