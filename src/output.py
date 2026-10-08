# output.py

from pathlib import Path

import pandas as pd


# ============================================================
# Output directory
# ============================================================

OUTPUT_DIR = Path(r"C:\Lab\brain_age_test_retest\output")

OUTPUT_FILES = {
    "ResNet": OUTPUT_DIR / "ResNet.xlsx",
    "SFCN_reg": OUTPUT_DIR / "SFCN_reg.xlsx",
    "SFCN_sm": OUTPUT_DIR / "SFCN_sm.xlsx",
}


# ============================================================
# Reset output Excel
# ============================================================

def reset_output_excel(model_group):
    """
    Full rerun 時刪除舊的 Excel。

    model_group:
        "ResNet"
        "SFCN_reg"
        "SFCN_sm"
    """
    if model_group not in OUTPUT_FILES:
        raise ValueError(
            f"未知 model_group: {model_group}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_FILES[model_group]

    if output_path.exists():
        output_path.unlink()

    return output_path


# ============================================================
# Build result DataFrame
# ============================================================

def build_result_dataframe(
    subject_ids,
    true_ages,
    pred_ages,
    filenames=None,
    memos=None,
):
    """
    不做任何額外計算。

    固定輸出：
        Subject ID
        True Age
        Predicted Age

    選擇性輸出：
        File Name
        Memo
    """

    data = {
        "Subject ID": subject_ids,
        "True Age": true_ages,
        "Predicted Age": pred_ages,
    }

    if filenames is not None:
        data["File Name"] = filenames

    if memos is not None:
        data["Memo"] = memos

    return pd.DataFrame(data)


# ============================================================
# Sheet name
# ============================================================

def make_sheet_name(
    dataset_name,
    modality=None,
    resnet_variant=None,
):
    """
    Examples:

    SFCN:
        Maclaren_T1WI
        MMRR_GMP
        OAS1_DGM

    ResNet:
        Maclaren_T1
        Maclaren_T1_SEX
        Maclaren_T1_B0
        Maclaren_T1_SEX_B0
    """

    if resnet_variant is not None:
        sheet_name = f"{dataset_name}_{resnet_variant}"

    elif modality is not None:
        sheet_name = f"{dataset_name}_{modality}"

    else:
        sheet_name = dataset_name

    # Excel sheet name cannot exceed 31 characters
    return sheet_name[:31]


# ============================================================
# Write result
# ============================================================

def write_result_sheet(
    model_group,
    sheet_name,
    result_df,
):
    """
    將 result_df 寫入指定 workbook 的 sheet。

    model_group:
        "ResNet"
        "SFCN_reg"
        "SFCN_sm"
    """

    if model_group not in OUTPUT_FILES:
        raise ValueError(
            f"未知 model_group: {model_group}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_FILES[model_group]

    if output_path.exists():

        with pd.ExcelWriter(
            output_path,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace",
        ) as writer:

            result_df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )

    else:

        with pd.ExcelWriter(
            output_path,
            engine="openpyxl",
            mode="w",
        ) as writer:

            result_df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )

    return output_path