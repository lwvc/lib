from pathlib import Path

LAB_ROOT = Path(r"C:\Lab")

DATA_ROOT = LAB_ROOT / "test_retest_dataset"
MODEL_ROOT = LAB_ROOT / "model_val_test"
OUTPUT_ROOT = LAB_ROOT / "brain_age_test_retest" / "output"

DATASETS = ["Maclaren", "MMRR", "OAS1"]

IMAGE_DIRS = {
    "T1WI": Path("ndf/rmi"),
    "GMP":  Path("df/mwp1"),
    "WMP":  Path("df/mwp2"),
    "DGM":  Path("ndf/rdgm"),
}


'''
from pathlib import Path

LAB_ROOT = Path(r"C:\Lab")

DATA_ROOT = LAB_ROOT / "test_retest_dataset"
MODEL_ROOT = LAB_ROOT / "model_val_test"
OUTPUT_ROOT = LAB_ROOT / "brain_age_test_retest" / "output"

DATASETS = ["Maclaren", "MMRR", "OAS1"]

IMAGE_DIRS = {
    "T1WI": Path("ndf/rmi"),
    "GMP":  Path("df/mwp1"),
    "WMP":  Path("df/mwp2"),

    # Deep grey matter
    # DGM image processing will be added later.
    "DGM":  Path("ndf/rdgm"),
}
---
def get_dataset_paths(dataset_name, modality):
    dataset_root = DATA_ROOT / dataset_name

    excel_path = dataset_root / f"{dataset_name}.xlsx"
    image_dir = dataset_root / IMAGE_DIRS[modality]

    return excel_path, image_dir
'''