from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
import copy
from loralib import layers as lora


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, is_lora: bool = False,
            lora_config: dict = None):
    """3x3 convolution with padding"""
    if is_lora:
        return lora.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            r=lora_config["r"], lora_alpha=lora_config["lora_alpha"], lora_dropout=lora_config["lora_dropout"],
            merge_weights=lora_config["merge_weights"],
            stride=stride, bias=False, groups=groups, padding=dilation, dilation=dilation)
    else:
        return nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            bias=False,
            dilation=dilation,
        )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, is_lora: bool = False, lora_config: dict = None):
    """1x1 convolution"""
    if is_lora:
        return lora.Conv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            r=lora_config["r"], lora_alpha=lora_config["lora_alpha"], lora_dropout=lora_config["lora_dropout"],
            merge_weights=lora_config["merge_weights"],
            stride=stride)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            is_lora: bool = False,
            lora_config: dict = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, is_lora=is_lora, lora_config=lora_config)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, is_lora=is_lora, lora_config=lora_config)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            is_lora: bool = False,
            lora_config: dict = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, is_lora=is_lora, lora_config=lora_config)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, is_lora=is_lora, lora_config=lora_config)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, is_lora=is_lora, lora_config=lora_config)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            lora_config: dict,
            is_lora: bool = False,
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,

    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        first_conv = lora_config["first_conv"]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # 去掉最大池化
        diff_layer_conv = lora_config["diff_layer_conv"]
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], is_lora=diff_layer_conv[0], lora_config=lora_config)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0],
                                       is_lora=diff_layer_conv[1], lora_config=lora_config)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1],
                                       is_lora=diff_layer_conv[2], lora_config=lora_config)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2],
                                       is_lora=diff_layer_conv[3], lora_config=lora_config)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
            is_lora: bool = False,
            lora_config: dict = None
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, is_lora=is_lora, lora_config=lora_config),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer,
                is_lora=is_lora,
                lora_config=lora_config
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    is_lora=is_lora,
                    lora_config=lora_config,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # x = self.fc2(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def load_weight(self, state_dict):
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        state_dict_tmp = copy.deepcopy(state_dict)
        for key, value in state_dict.items():
            if "conv1.weight" in key and "conv1.weight" != key:
                state_dict_tmp.update({key.replace("conv1.weight", "conv1.conv.weight"): state_dict_tmp.pop(key)})
            elif "conv2.weight" in key:
                state_dict_tmp.update({key.replace("conv2.weight", "conv2.conv.weight"): state_dict_tmp.pop(key)})
            elif "conv3.weight" in key:
                state_dict_tmp.update({key.replace("conv3.weight", "conv3.conv.weight"): state_dict_tmp.pop(key)})
            elif "conv1.bias" in key:
                state_dict_tmp.update({key.replace("conv1.bias", "conv1.conv.bias"): state_dict_tmp.pop(key)})
            elif "conv2.bias" in key:
                state_dict_tmp.update({key.replace("conv2.bias", "conv2.conv.bias"): state_dict_tmp.pop(key)})
            elif "conv3.bias" in key:
                state_dict_tmp.update({key.replace("conv3.bias", "conv3.conv.bias"): state_dict_tmp.pop(key)})
            elif "downsample.0.weight" in key:
                state_dict_tmp.update(
                    {key.replace("downsample.0.weight", "downsample.0.conv.weight"): state_dict_tmp.pop(key)})
            for key, value in self.state_dict().items():
                if "lora_" in key:
                    state_dict_tmp[key] = value
                elif "bias" in key:
                    state_dict_tmp[key] = value
                elif "conv1.weight" == key:
                    state_dict_tmp[key] = value
                elif "fc.weight" == key:
                    state_dict_tmp[key] = value
        self.load_state_dict(state_dict_tmp)


def ResNet18(is_lora, lora_config, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], is_lora=is_lora, lora_config=lora_config, **kwargs)


def ResNet34(is_lora, lora_config, **kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], is_lora=is_lora, lora_config=lora_config, **kwargs)


def ResNet50(is_lora, lora_config, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], is_lora=is_lora, lora_config=lora_config, **kwargs)


def ResNet101(is_lora, lora_config, **kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], is_lora=is_lora, lora_config=lora_config, **kwargs)

