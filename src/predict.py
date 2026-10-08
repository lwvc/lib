# predict.py

import numpy as np
import torch


# ============================================================
# SFCN Softmax
#
# Original predicted-age calculation preserved:
# model -> log probability -> exp -> expected age
# ============================================================

def predict_sfcn_softmax(model, images, device):
    images = images.to(device)

    outputs = (
        model(images)[0]
        .squeeze(-1)
        .squeeze(-1)
        .squeeze(-1)
    )

    probs = outputs.exp()

    bins = np.linspace(5, 100, 96)

    bin_centers = [
        (bins[i] + bins[i + 1]) / 2
        for i in range(len(bins) - 1)
    ]

    age_bins_center = torch.tensor(
        bin_centers,
        dtype=torch.float32,
    ).to(device)

    pred_age = torch.sum(
        probs * age_bins_center,
        dim=1,
    )

    return pred_age


# ============================================================
# SFCN Regression
#
# Original predicted-age calculation preserved:
# model output = predicted age
# ============================================================

def predict_sfcn_regression(model, images, device):
    images = (
        images
        .detach()
        .clone()
        .float()
        .to(
            device,
            non_blocking=True,
        )
    )

    outputs = model(images).view(-1)

    return outputs


# ============================================================
# ResNet
#
# Original feature-input logic preserved.
#
# T1              -> no extra feature
# T1 + Sex        -> sex
# T1 + B0         -> scanner
# T1 + Sex + B0   -> sex, scanner
# ============================================================

def predict_resnet(
    model,
    images,
    device,
    use_sex=False,
    use_b0=False,
    sex=None,
    scanner=None,
):
    images = images.to(device)

    feats = []

    # Keep original order:
    # 1. Sex
    # 2. B0 / Scanner
    if use_sex:
        if sex is None:
            raise ValueError(
                "此 ResNet model 需要 sex input。"
            )

        feats.append(
            sex.to(device)
        )

    if use_b0:
        if scanner is None:
            raise ValueError(
                "此 ResNet model 需要 B0/scanner input。"
            )

        feats.append(
            scanner.to(device)
        )

    extra_feat = (
        torch.cat(
            feats,
            dim=1,
        )
        if feats
        else None
    )

    outputs = (
        model(
            images,
            extra_feat,
        )
        .cpu()
        .numpy()
        .flatten()
    )

    return outputs


# ============================================================
# Unified prediction dispatcher
#
# Only selects the correct original prediction method.
# Does not modify model architecture or calculations.
# ============================================================

def predict_batch(
    model,
    batch,
    model_config,
    device,
):
    family = model_config["family"]
    task = model_config["task"]

    # --------------------------------------------------------
    # SFCN
    # --------------------------------------------------------
    if family == "sfcn":

        images = batch["image"]

        if task == "softmax":
            return predict_sfcn_softmax(
                model=model,
                images=images,
                device=device,
            )

        if task == "regression":
            return predict_sfcn_regression(
                model=model,
                images=images,
                device=device,
            )

        raise ValueError(
            f"未知 SFCN task: {task}"
        )

    # --------------------------------------------------------
    # ResNet
    # --------------------------------------------------------
    if family == "resnet":

        return predict_resnet(
            model=model,
            images=batch["image"],
            device=device,
            use_sex=model_config["use_sex"],
            use_b0=model_config["use_b0"],
            sex=batch.get("sex"),
            scanner=batch.get("scanner"),
        )

    raise ValueError(
        f"未知 model family: {family}"
    )