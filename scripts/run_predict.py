# run_predict.py

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from src.dataset import BrainAgeDataset
from src.model_registry import get_model_config

from src.models.sfcn import (
    SFCNSoftmax,
    SFCNRegression,
)

from src.models.resnet import (
    ResBlock,
    NetResBlock,
    RESNET_MODEL,
)

from src.predict import predict_batch


def _alias_resnet_for_unpickle():
    # ResNet checkpoints were saved with torch.save(full model) where
    # classes lived in __main__. Unpickling looks up __main__.RESNET_MODEL
    # / NetResBlock / ResBlock, so alias them to the real implementations.
    import __main__ as _main

    for _name, _cls in (
        ("ResBlock", ResBlock),
        ("NetResBlock", NetResBlock),
        ("RESNET_MODEL", RESNET_MODEL),
    ):
        if not hasattr(_main, _name):
            setattr(_main, _name, _cls)

from src.output import (
    reset_output_excel,
    build_result_dataframe,
    make_sheet_name,
    write_result_sheet,
)


# ============================================================
# Basic settings
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DATASETS = [
    "Maclaren",
    "MMRR",
    "OAS1",
]


# ============================================================
# Models to run
#
# DGM is temporarily disabled because additional
# Deep Grey Matter image processing is still required.
# ============================================================

SFCN_SM_MODELS = [
    "sfcn_sm_t1",
    "sfcn_sm_gmp",
    "sfcn_sm_wmp",

    # Deep grey matter
    # "sfcn_sm_dgm",
]

SFCN_REG_MODELS = [
    "sfcn_reg_t1",
    "sfcn_reg_gmp",
    "sfcn_reg_wmp",

    # Deep grey matter
    # "sfcn_reg_dgm",
]

RESNET_MODELS = [
    "resnet_t1",
    "resnet_t1_sex",
    "resnet_t1_b0",
    "resnet_t1_sex_b0",
]


# ============================================================
# Original ResNet test sampler
#
# Keep test-time sampler behavior unchanged:
# TestBalancedAgeBatchSampler(batch_size=1).
# ============================================================

class TestBalancedAgeBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size=1,
    ):
        self.dataset = dataset
        self.batch_size = batch_size

        ages = (
            dataset.df["Age"]
            .to_numpy()
        )

        quantiles = np.linspace(
            0,
            1,
            batch_size + 1,
        )

        bins = np.quantile(
            ages,
            quantiles,
        )

        self.age_bins = [
            (
                bins[i],
                bins[i + 1],
            )
            for i in range(
                len(bins) - 1
            )
        ]

        self.age_bins[-1] = (
            self.age_bins[-1][0],
            self.age_bins[-1][1] + 0.1,
        )

        groups = {
            b: []
            for b in self.age_bins
        }

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

                    batch.append(
                        groups[b].pop(0)
                    )

                if len(batch) == self.batch_size:
                    break

            if batch:
                batches.append(batch)

        self.batches = batches

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)


# ============================================================
# Build model
# ============================================================

def build_model(model_config):

    family = model_config["family"]
    task = model_config["task"]

    # --------------------------------------------------------
    # SFCN
    # --------------------------------------------------------
    if family == "sfcn":

        if task == "softmax":

            model = SFCNSoftmax(
                output_dim=model_config["output_dim"],
                avg_shape=model_config["avg_shape"],
            )

        elif task == "regression":

            model = SFCNRegression(
                prediction_range=(
                    model_config["prediction_range"]
                ),
                avg_shape=model_config["avg_shape"],
            )

        else:

            raise ValueError(
                f"未知 SFCN task: {task}"
            )

        model = model.to(DEVICE)

        # Keep original SFCN checkpoint loading behavior.
        state_dict = torch.load(
            model_config["checkpoint"],
            map_location=DEVICE,
            weights_only=True,
        )

        model.load_state_dict(
            state_dict,
            strict=True,
        )

        model.eval()

        return model

    # --------------------------------------------------------
    # ResNet
    # --------------------------------------------------------
    if family == "resnet":

        _alias_resnet_for_unpickle()

        model_type = model_config["model_type"]

        if model_type == "cat":

            model = RESNET_MODEL(
                model_type=model_type,
                cat_dim=model_config["cat_dim"],
            )

        else:

            model = RESNET_MODEL(
                model_type=model_type,
            )

        model = model.to(DEVICE)

        # Keep original ResNet checkpoint loading behavior.
        pretrained = torch.load(
            model_config["checkpoint"],
            map_location=DEVICE,
            weights_only=False,
        )

        state_dict = (
            pretrained.state_dict()
            if hasattr(
                pretrained,
                "state_dict",
            )
            else pretrained
        )

        model.load_state_dict(
            state_dict,
            strict=True,
        )

        model.eval()

        return model

    raise ValueError(
        f"未知 model family: {family}"
    )


# ============================================================
# Build DataLoader
# ============================================================

def build_loader(
    dataset_name,
    model_config,
):

    family = model_config["family"]
    modality = model_config["modality"]

    # --------------------------------------------------------
    # SFCN
    # --------------------------------------------------------
    if family == "sfcn":

        dataset = BrainAgeDataset(
            dataset_name=dataset_name,
            modality=modality,
            crop_coords=model_config["crop"],
            return_metadata=True,
        )

        loader = DataLoader(
            dataset,
            batch_size=4,  # 原為 4 
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        return dataset, loader

    # --------------------------------------------------------
    # ResNet
    #
    # No crop.
    # Keep original test sampler behavior.
    # --------------------------------------------------------
    if family == "resnet":

        dataset = BrainAgeDataset(
            dataset_name=dataset_name,
            modality="T1WI",
            crop_coords=None,
            return_metadata=True,
        )

        sampler = TestBalancedAgeBatchSampler(
            dataset,
            batch_size=1,
        )

        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=0,
        )

        return dataset, loader

    raise ValueError(
        f"未知 model family: {family}"
    )


# ============================================================
# Evaluate one model on one dataset
# ============================================================

def run_one(
    dataset_name,
    model_name,
    model_group,
):

    config = get_model_config(
        model_name
    )

    print("\n" + "=" * 70)
    print(f"Dataset : {dataset_name}")
    print(f"Model   : {model_name}")
    print(f"Device  : {DEVICE}")
    print("=" * 70)

    model = build_model(
        config
    )

    dataset, loader = build_loader(
        dataset_name,
        config,
    )

    subject_ids = []
    true_ages = []
    pred_ages = []
    filenames = []
    memos = []

    has_memo = (
        "Memo" in dataset.df.columns
    )

    model.eval()

    with torch.no_grad():

        for batch in loader:

            # --------------------------------------------
            # Prediction
            #
            # All original predicted-age calculations
            # are kept inside predict.py.
            # --------------------------------------------

            predictions = predict_batch(
                model=model,
                batch=batch,
                model_config=config,
                device=DEVICE,
            )

            if torch.is_tensor(predictions):

                predictions = (
                    predictions
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .tolist()
                )

            else:

                predictions = (
                    np.asarray(predictions)
                    .reshape(-1)
                    .tolist()
                )

            # --------------------------------------------
            # Metadata only
            # --------------------------------------------

            batch_subject_ids = [
                str(x)
                for x in batch["subject_id"]
            ]

            batch_filenames = [
                str(x)
                for x in batch["filename"]
            ]

            batch_true_ages = (
                batch["age"]
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
                .tolist()
            )

            subject_ids.extend(
                batch_subject_ids
            )

            filenames.extend(
                batch_filenames
            )

            true_ages.extend(
                batch_true_ages
            )

            pred_ages.extend(
                predictions
            )

            # Memo is optional.
            if has_memo:

                if "memo" in batch:

                    memos.extend(
                        [
                            str(x)
                            for x in batch["memo"]
                        ]
                    )

    # --------------------------------------------------------
    # Output
    # No MAE / PAD / ICC calculation here.
    # --------------------------------------------------------

    result_df = build_result_dataframe(
        subject_ids=subject_ids,
        true_ages=true_ages,
        pred_ages=pred_ages,
        filenames=filenames,
        memos=(
            memos
            if len(memos) == len(subject_ids)
            else None
        ),
    )

    # --------------------------------------------------------
    # Sheet name
    # --------------------------------------------------------

    if config["family"] == "resnet":

        variant_map = {
            "resnet_t1":
                "T1",

            "resnet_t1_sex":
                "T1_SEX",

            "resnet_t1_b0":
                "T1_B0",

            "resnet_t1_sex_b0":
                "T1_SEX_B0",
        }

        sheet_name = make_sheet_name(
            dataset_name=dataset_name,
            resnet_variant=(
                variant_map[model_name]
            ),
        )

    else:

        sheet_name = make_sheet_name(
            dataset_name=dataset_name,
            modality=config["modality"],
        )

    output_path = write_result_sheet(
        model_group=model_group,
        sheet_name=sheet_name,
        result_df=result_df,
    )

    print(
        f"完成: {sheet_name}"
    )

    print(
        f"樣本數: {len(result_df)}"
    )

    print(
        f"輸出: {output_path}"
    )


# ============================================================
# Run model group
# ============================================================

def run_model_group(
    model_group,
    model_names,
):

    print("\n" + "#" * 70)
    print(f"開始執行 {model_group}")
    print("#" * 70)

    reset_output_excel(
        model_group
    )

    for model_name in model_names:

        config = get_model_config(
            model_name
        )

        checkpoint = config["checkpoint"]

        if not checkpoint.exists():

            raise FileNotFoundError(
                f"找不到模型權重:\n"
                f"{checkpoint}"
            )

        for dataset_name in DATASETS:

            run_one(
                dataset_name=dataset_name,
                model_name=model_name,
                model_group=model_group,
            )


# ============================================================
# Main
# ============================================================

def main():

    print(
        f"目前使用裝置: {DEVICE}"
    )

    # --------------------------------------------------------
    # SFCN Softmax
    # --------------------------------------------------------

    run_model_group(
        model_group="SFCN_sm",
        model_names=SFCN_SM_MODELS,
    )

    # --------------------------------------------------------
    # SFCN Regression
    # --------------------------------------------------------

    run_model_group(
        model_group="SFCN_reg",
        model_names=SFCN_REG_MODELS,
    )

    # --------------------------------------------------------
    # ResNet
    # --------------------------------------------------------

    run_model_group(
        model_group="ResNet",
        model_names=RESNET_MODELS,
    )

    print(
        "\n全部 prediction 完成。"
    )


if __name__ == "__main__":
    main()