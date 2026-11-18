import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.networks.dpt.base_model import BaseModel
from src.networks.dpt.blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)

# from .base_model import BaseModel
# from .blocks import (
#     FeatureFusionBlock,
#     FeatureFusionBlock_custom,
#     Interpolate,
#     _make_encoder,
#     forward_vit,
# )


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class DPT(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):
        super(DPT, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }
        self.backbone_name = backbone

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            False,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head

    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return out


class DPTDepthModel(DPT):
    def __init__(
        self, path=None, non_negative=True, scale=1.0, shift=0.0, invert=False, **kwargs
    ):
        features = kwargs["features"] if "features" in kwargs else 256

        self.scale = scale
        self.shift = shift
        self.invert = invert

        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)

    def forward(self, x):
        inv_depth = super().forward(x).squeeze(dim=1)

        if self.invert:
            depth = self.scale * inv_depth + self.shift
            depth[depth < 1e-8] = 1e-8
            depth = 1.0 / depth
            return depth
        else:
            return inv_depth


class DPTSegmentationModel(DPT):
    def __init__(self, num_classes, path=None, **kwargs):
        features = kwargs["features"] if "features" in kwargs else 256

        kwargs["use_bn"] = True

        head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, **kwargs)

        self.auxlayer = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
        )

        if path is not None:
            self.load(path)


class DPTRainPredictionModel(DPT):
    def __init__(
        self,
        path=None,
        non_negative=False,
        scale=1.0,
        shift=0.0,
        invert=False,
        in_channels=12,
        **kwargs,
    ):
        features = kwargs["features"] if "features" in kwargs else 256

        self.scale = scale
        self.shift = shift
        self.invert = invert

        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

        super().__init__(head, **kwargs)

        # change the first patch layer
        if self.backbone_name == "vitb_rn50_384":
            from timm.layers.std_conv import StdConv2dSame

            self.pretrained.model.patch_embed.backbone.stem.conv = StdConv2dSame(
                in_channels, 64, kernel_size=7, stride=2, bias=False
            )
        # todo: add more first patch layers compatible with other backbones
        else:
            raise NotImplementedError(
                f"Backbone {self.backbone_name} not implemented for train prediction model."
            )

        if path is not None:
            self.load(path)

        self.apply(self.init_model)

    def init_model(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def conditions_drop(self, x_list):
        """
        Randomly drop some conditions (e.g., satellite, radar, rain) during training.
        """
        if self.training:
            ...

        return x_list

    def img_time_to_channel(self, x_list):
        # warn: simple strategy: maybe wrong in time spective !
        x_list_new = []
        for x in x_list:
            if x.dim() == 4:
                x_list_new.append(x)
            elif x.dim() == 5:
                x = rearrange(x, "b c t h w -> b (c t) h w")
                x_list_new.append(x)
            else:
                raise ValueError(f"Unsupported tensor dimension: {x.dim()}")

        return x_list_new

    def output_channel_to_time(self, output):
        return rearrange(
            output, "b (c t) h w -> b c t h w", c=1
        )  #  rain map channel is 1

    def forward(self, *x_list):
        assert len(x_list) == 3, "Radar, Satellite, and Rain data are required"
        x_list = self.conditions_drop(x_list)  # CFG drop?
        x_list = self.img_time_to_channel(x_list)

        # TODO: not early-fusion, other strategy?
        x = torch.cat(x_list, dim=1)  # Concatenate along channel dimension

        rain = super().forward(x)
        if self.invert:
            rain = self.scale * rain + self.shift
            rain[rain < 1e-8] = 1e-8
            rain = 1.0 / rain

        # print(f"rain max: {rain.max()}")
        return self.output_channel_to_time(rain)


if __name__ == "__main__":
    device = "cuda:0"
    model = DPTRainPredictionModel(
        scale=1,
        shift=False,
    ).to(device)
    model.eval()

    # import fvcore.nn as fnn
    # print(fnn.parameter_count_table(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    x1 = torch.randn(1, 1, 384, 384).to(device)  # Radar
    x2 = torch.randn(1, 10, 384, 384).to(device)  # Satellite
    x3 = torch.randn(1, 1, 384, 384).to(device)  # Rain

    y = model(x1, x2, x3)
    print(y.shape)  # Should be [1, 1, 384, 384]

    optimizer.zero_grad()
    y.sum().backward()
    optimizer.step()

    import time

    time.sleep(20)
