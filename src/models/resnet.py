# models/resnet.py

import torch


# ============================================================
# ResNet block
# Original architecture preserved.
# ============================================================

class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = torch.nn.Conv3d(
            in_channels,
            out_channels,
            3,
            1,
            1,
        )

        self.batchnorm = torch.nn.BatchNorm3d(
            out_channels,
            affine=True,
            track_running_stats=False,
        )

        self.ELU = torch.nn.ELU()

        self.conv2 = torch.nn.Conv3d(
            out_channels,
            out_channels,
            3,
            1,
            1,
        )

        self.conv_shortcut = torch.nn.Conv3d(
            in_channels,
            out_channels,
            1,
            1,
            0,
        )

        self.maxpooling = torch.nn.MaxPool3d(
            2,
            2,
            ceil_mode=False,
        )

    def forward(self, x):

        x_ = self.conv_shortcut(x)

        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.ELU(x)

        x = self.conv2(x)
        x = self.batchnorm(x)

        x = self.ELU(x + x_)

        return x


# ============================================================
# ResNet feature extractor
# Original architecture preserved.
# ============================================================

class NetResBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.resblock1 = ResBlock(1, 8)
        self.resblock2 = ResBlock(8, 16)
        self.resblock3 = ResBlock(16, 32)
        self.resblock4 = ResBlock(32, 64)
        self.resblock5 = ResBlock(64, 128)
        self.resblock6 = ResBlock(128, 256)

        self.maxpooling1 = torch.nn.MaxPool3d(
            2,
            2,
            1,
            ceil_mode=False,
        )

        self.maxpooling2 = torch.nn.MaxPool3d(
            2,
            2,
            (1, 0, 1),
            ceil_mode=False,
        )

    def forward(self, x):

        x = self.resblock1(x)
        x = self.maxpooling1(x)

        x = self.resblock2(x)
        x = self.maxpooling1(x)

        x = self.resblock3(x)
        x = self.maxpooling2(x)

        x = self.resblock4(x)
        x = self.maxpooling1(x)

        x = self.resblock5(x)
        x = self.maxpooling1(x)

        x = self.resblock6(x)
        x = self.maxpooling2(x)

        return x


# ============================================================
# Brain-age ResNet
#
# model_type:
#   "no_cat" -> T1 only
#   "cat"    -> T1 + additional feature(s)
#
# Do not change model architecture or feature concatenation.
# ============================================================

class RESNET_MODEL(torch.nn.Module):
    def __init__(
        self,
        model_type="no_cat",
        cat_dim=257,
    ):
        super().__init__()

        self.RES = NetResBlock()

        self.flatten = torch.nn.Flatten()

        self.fc1 = torch.nn.Linear(
            16384,
            256,
        )

        self.ELU = torch.nn.ELU()

        self.dropout = torch.nn.Dropout(
            p=0.2,
        )

        self.model_type = model_type

        # Keep parameter names aligned with original pretrained weights.
        self.fc2_for_no_cat = torch.nn.Linear(
            256,
            1,
        )

        self.fc2_for_cat = torch.nn.Linear(
            cat_dim,
            1,
        )

    def forward(self, x, extra_feat=None):

        x = self.RES(x)

        x = self.flatten(x)

        x = self.fc1(x)

        x = self.ELU(x)

        x = self.dropout(x)

        if self.model_type == "cat":

            x = torch.cat(
                (x, extra_feat),
                dim=1,
            )

            x = self.fc2_for_cat(x)

        else:

            x = self.fc2_for_no_cat(x)

        return x