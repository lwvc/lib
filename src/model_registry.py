# model_registry.py

from pathlib import Path

import numpy as np


LAB_ROOT = Path(r"C:\Lab")

MODEL_ROOT = LAB_ROOT / "model_val_test_train"

SFCN_ROOT = (
    MODEL_ROOT
    / "SFCN"
    / "model"
)

RESNET_ROOT = (
    MODEL_ROOT
    / "ResNet"
)


# ============================================================
# SFCN spatial settings
# ============================================================

GLOBAL_CROP = (
    np.array([11, 12, 10]),
    np.array([181, 221, 177]),
)

DGM_CROP = (
    np.array([52, 77, 50]),
    np.array([140, 168, 122]),
)

GLOBAL_AVG_SHAPE = (5, 6, 5)
DGM_AVG_SHAPE = (2, 2, 2)


# ============================================================
# Model registry
# ============================================================

MODEL_REGISTRY = {

    # --------------------------------------------------------
    # SFCN - Softmax
    # --------------------------------------------------------

    "sfcn_sm_t1": {
        "family": "sfcn",
        "task": "softmax",
        "modality": "T1WI",

        "checkpoint": (
            SFCN_ROOT
            / "softmax"
            / "pretrained"
            / "pretrain_globalFOV_SM_T1_withcc_v2_20260214.pt"
        ),

        "crop": GLOBAL_CROP,
        "avg_shape": GLOBAL_AVG_SHAPE,

        "output_dim": 95,
    },

    "sfcn_sm_gmp": {
        "family": "sfcn",
        "task": "softmax",
        "modality": "GMP",

        "checkpoint": (
            SFCN_ROOT
            / "softmax"
            / "pretrained"
            / "pretrain_globalFOV_SM_GMP_withcc_v1_20260301.pt"
        ),

        "crop": GLOBAL_CROP,
        "avg_shape": GLOBAL_AVG_SHAPE,

        "output_dim": 95,
    },

    "sfcn_sm_wmp": {
        "family": "sfcn",
        "task": "softmax",
        "modality": "WMP",

        "checkpoint": (
            SFCN_ROOT
            / "softmax"
            / "pretrained"
            / "pretrain_globalFOV_SM_WMP_withcc_v1_20260304.pt"
        ),

        "crop": GLOBAL_CROP,
        "avg_shape": GLOBAL_AVG_SHAPE,

        "output_dim": 95,
    },

    "sfcn_sm_dgm": {
        "family": "sfcn",
        "task": "softmax",
        "modality": "DGM",

        "checkpoint": (
            SFCN_ROOT
            / "softmax"
            / "pretrained"
            / "pretrain_regionalFOV_SM_DGM_withcc_v2_20260305.pt"
        ),

        # Deep grey matter
        # DGM requires additional image processing.
        "crop": DGM_CROP,
        "avg_shape": DGM_AVG_SHAPE,

        "output_dim": 95,
    },


    # --------------------------------------------------------
    # SFCN - Regression
    # --------------------------------------------------------

    "sfcn_reg_t1": {
        "family": "sfcn",
        "task": "regression",
        "modality": "T1WI",

        "checkpoint": (
            SFCN_ROOT
            / "regression"
            / "pretrained"
            / "pretrain_globalFOV_REG_T1_withcc_v2_20260226.pt"
        ),

        "crop": GLOBAL_CROP,
        "avg_shape": GLOBAL_AVG_SHAPE,

        "prediction_range": (5, 100),
    },

    "sfcn_reg_gmp": {
        "family": "sfcn",
        "task": "regression",
        "modality": "GMP",

        "checkpoint": (
            SFCN_ROOT
            / "regression"
            / "pretrained"
            / "pretrain_globalFOV_REG_GMP_withcc_v2_20260314.pt"
        ),

        "crop": GLOBAL_CROP,
        "avg_shape": GLOBAL_AVG_SHAPE,

        "prediction_range": (5, 100),
    },

    "sfcn_reg_wmp": {
        "family": "sfcn",
        "task": "regression",
        "modality": "WMP",

        "checkpoint": (
            SFCN_ROOT
            / "regression"
            / "pretrained"
            / "pretrain_globalFOV_REG_WMP_withcc_v1_20260306.pt"
        ),

        "crop": GLOBAL_CROP,
        "avg_shape": GLOBAL_AVG_SHAPE,

        "prediction_range": (5, 100),
    },

    "sfcn_reg_dgm": {
        "family": "sfcn",
        "task": "regression",
        "modality": "DGM",

        "checkpoint": (
            SFCN_ROOT
            / "softmax"
            / "pretrained"
            / "pretrain_regionalFOV_REG_DGM_withcc_v1_20260227.pt"
        ),

        # Deep grey matter
        # DGM requires additional image processing.
        "crop": DGM_CROP,
        "avg_shape": DGM_AVG_SHAPE,

        "prediction_range": (5, 100),
    },


    # --------------------------------------------------------
    # ResNet
    #
    # IMPORTANT:
    # No crop is applied here.
    # Original model input settings are preserved.
    # --------------------------------------------------------

    "resnet_t1_b0": {
        "family": "resnet",
        "task": "regression",
        "modality": "T1WI",

        "checkpoint": (
            RESNET_ROOT
            / "pretrained"
            / "model_20250620_095129_948.pt"
        ),

        "crop": None,
        "avg_shape": None,

        "use_sex": False,
        "use_b0": True,

        "model_type": "cat",
        "cat_dim": 257,
    },

    "resnet_t1": {
        "family": "resnet",
        "task": "regression",
        "modality": "T1WI",

        "checkpoint": (
            RESNET_ROOT
            / "pretrained"
            / "model_20250624_095907_881.pt"
        ),

        "crop": None,
        "avg_shape": None,

        "use_sex": False,
        "use_b0": False,

        "model_type": "no_cat",
        "cat_dim": None,
    },

    "resnet_t1_sex_b0": {
        "family": "resnet",
        "task": "regression",
        "modality": "T1WI",

        "checkpoint": (
            RESNET_ROOT
            / "pretrained"
            / "model_20250628_092154_757.pt"
        ),

        "crop": None,
        "avg_shape": None,

        "use_sex": True,
        "use_b0": True,

        "model_type": "cat",
        "cat_dim": 258,
    },

    "resnet_t1_sex": {
        "family": "resnet",
        "task": "regression",
        "modality": "T1WI",

        "checkpoint": (
            RESNET_ROOT
            / "pretrained"
            / "model_20250705_182002_540.pt"
        ),

        "crop": None,
        "avg_shape": None,

        "use_sex": True,
        "use_b0": False,

        "model_type": "cat",
        "cat_dim": 257,
    },
}


# ============================================================
# Helper
# ============================================================

def get_model_config(model_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"未知模型: {model_name}\n"
            f"可用模型: {list(MODEL_REGISTRY.keys())}"
        )

    return MODEL_REGISTRY[model_name]