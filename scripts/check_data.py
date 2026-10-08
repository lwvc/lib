# check_data.py

from pathlib import Path

import nibabel as nib
import pandas as pd

from src.dataset import (
    DATASET_FILES,
    MODALITY_CONFIG,
)


DATASETS = [
    "Maclaren",
    "MMRR",
    "OAS1",
]


# ============================================================
# Check one dataset + modality
# ============================================================

def check_dataset(
    dataset_name,
    modality,
    check_image_shape=True,
):
    excel_path = DATASET_FILES[dataset_name]

    config = MODALITY_CONFIG[modality]

    image_dir = (
        excel_path.parent
        / config["folder"]
    )

    filename_col = config["filename_col"]

    print("\n" + "=" * 70)
    print(f"Dataset  : {dataset_name}")
    print(f"Modality : {modality}")
    print(f"Excel    : {excel_path}")
    print(f"Images   : {image_dir}")
    print("=" * 70)

    # --------------------------------------------------------
    # Excel
    # --------------------------------------------------------
    if not excel_path.exists():
        print("[ERROR] Excel 不存在")
        return

    df = pd.read_excel(excel_path)

    required_cols = [
        "Subject_ID",
        "Age",
        filename_col,
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        print(
            f"[ERROR] 缺少欄位: {missing_cols}"
        )
        return

    print(f"Excel samples : {len(df)}")

    # --------------------------------------------------------
    # Missing metadata
    # --------------------------------------------------------
    missing_subject = (
        df["Subject_ID"]
        .isna()
        .sum()
    )

    missing_age = (
        df["Age"]
        .isna()
        .sum()
    )

    missing_filename = (
        df[filename_col]
        .isna()
        .sum()
    )

    print(
        f"Missing Subject_ID : {missing_subject}"
    )

    print(
        f"Missing Age        : {missing_age}"
    )

    print(
        f"Missing Filename   : {missing_filename}"
    )

    # --------------------------------------------------------
    # Duplicate filenames
    # --------------------------------------------------------
    valid_filename = (
        df[filename_col]
        .dropna()
        .astype(str)
        .str.strip()
    )

    duplicate_mask = (
        valid_filename
        .duplicated(
            keep=False,
        )
    )

    duplicated = (
        valid_filename[
            duplicate_mask
        ]
        .tolist()
    )

    print(
        f"Duplicate Filename : {len(duplicated)}"
    )

    if duplicated:
        for filename in duplicated:
            print(
                f"  [DUPLICATE] {filename}"
            )

    # --------------------------------------------------------
    # Check image files
    # --------------------------------------------------------
    missing_files = []
    unreadable_files = []
    shape_count = {}

    checked = 0

    for _, row in df.iterrows():

        value = row[filename_col]

        if pd.isna(value):
            continue

        filename = str(value).strip()

        if not filename.endswith(
            (".nii", ".nii.gz")
        ):
            filename += ".nii"

        image_path = (
            image_dir
            / filename
        )

        if not image_path.exists():

            missing_files.append(
                str(image_path)
            )

            continue

        checked += 1

        if check_image_shape:

            try:
                img = nib.load(
                    image_path
                )

                shape = tuple(
                    img.shape
                )

                shape_count[shape] = (
                    shape_count.get(
                        shape,
                        0,
                    )
                    + 1
                )

            except Exception as e:

                unreadable_files.append(
                    (
                        str(image_path),
                        str(e),
                    )
                )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print(
        f"Images found       : {checked}"
    )

    print(
        f"Images missing     : {len(missing_files)}"
    )

    print(
        f"Images unreadable  : {len(unreadable_files)}"
    )

    if check_image_shape:

        print(
            "Image shapes:"
        )

        for shape, count in shape_count.items():

            print(
                f"  {shape}: {count}"
            )

    if missing_files:

        print(
            "\nMissing files:"
        )

        for path in missing_files:

            print(
                f"  [MISSING] {path}"
            )

    if unreadable_files:

        print(
            "\nUnreadable files:"
        )

        for path, error in unreadable_files:

            print(
                f"  [ERROR] {path}"
            )

            print(
                f"          {error}"
            )

    print(
        "\nCheck finished."
    )


# ============================================================
# Main
# ============================================================

def main():

    modalities = [
        "T1WI",
        "GMP",
        "WMP",

        # Deep grey matter
        # Add "DGM" after DGM image processing
        # and dataset configuration are finalized.
    ]

    for dataset_name in DATASETS:

        for modality in modalities:

            check_dataset(
                dataset_name,
                modality,
                check_image_shape=True,
            )


if __name__ == "__main__":
    main()