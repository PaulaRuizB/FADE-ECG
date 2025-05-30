import torch
import torch.nn as nn
from typing import Any, Callable, List, Optional, Type, Union, Tuple, Dict, TypeVar
from torch import Tensor
from types import FunctionType
import collections
from itertools import repeat
from functools import partial
from enum import Enum
import numpy as np
from timm.layers.drop import DropPath
torch.seed()

import random
random.seed(32)

np.random.seed(32)

def _log_api_usage_once(obj: Any) -> None:
    """
    Logs API usage(module and name) within an organization.
    In a large ecosystem, it's often useful to track the PyTorch and
    TorchVision APIs usage. This API provides the similar functionality to the
    logging module in the Python stdlib. It can be used for debugging purpose
    to log which methods are used and by default it is inactive, unless the user
    manually subscribes a logger via the `SetAPIUsageLogger method <https://github.com/pytorch/pytorch/blob/eb3b9fe719b21fae13c7a7cf3253f970290a573e/c10/util/Logging.cpp#L114>`_.
    Please note it is triggered only once for the same API call within a process.
    It does not collect any data from open-source users since it is no-op by default.
    For more information, please refer to
    * PyTorch note: https://pytorch.org/docs/stable/notes/large_scale_deployments.html#api-usage-logging;
    * Logging policy: https://github.com/pytorch/vision/issues/5052;
    Args:
        obj (class instance or method): an object to extract info from.
    """
    module = obj.__module__
    if not module.startswith("torchvision"):
        module = f"torchvision.internal.{module}"
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{module}.{name}")


V = TypeVar("V")


def _ovewrite_named_param(kwargs: Dict[str, Any], param: str, new_value: V) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    else:
        kwargs[param] = new_value


def _make_ntuple(x: Any, n: int) -> Tuple[Any, ...]:
    """
    Make n-tuple from input x. If x is an iterable, then we just convert it to tuple.
    Otherwise, we will make a tuple of length n, all with value of x.
    reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/utils.py#L8
    Args:
        x (Any): input value
        n (int): length of the resulting tuple
    """
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
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


def conv3x1(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=(3, 1),
        stride=stride,
        padding=(1, 0),
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=(1, 3),
        stride=stride,
        padding=(0, 1),
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1d_1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv1d_3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv1d:
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, padding=1, stride=stride, bias=False, groups=groups, dilation=dilation)

def conv1d_16(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv1d:
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=17, padding=8, stride=stride, bias=False, groups=groups, dilation=dilation)


class WeightsEnum(Enum):
    """
    This class is the parent class of all model weights. Each model building method receives an optional `weights`
    parameter with its associated pre-trained weights. It inherits from `Enum` and its values should be of type
    `Weights`.
    Args:
        value (Weights): The data class entry with the weight information.
    """

    @classmethod
    def verify(cls, obj: Any) -> Any:
        if obj is not None:
            if type(obj) is str:
                obj = cls[obj.replace(cls.__name__ + ".", "")]
            elif not isinstance(obj, cls):
                raise TypeError(
                    f"Invalid Weight class provided; expected {cls.__name__} but received {obj.__class__.__name__}."
                )
        return obj

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self._name_}"

    @property
    def url(self):
        return self.value.url

    @property
    def transforms(self):
        return self.value.transforms

    @property
    def meta(self):
        return self.value.meta


class Weights:
    """
    This class is used to group important attributes associated with the pre-trained weights.
    Args:
        url (str): The location where we find the weights.
        transforms (Callable): A callable that constructs the preprocessing method (or validation preset transforms)
            needed to use the model. The reason we attach a constructor method rather than an already constructed
            object is because the specific object might have memory and thus we want to delay initialization until
            needed.
        meta (Dict[str, Any]): Stores meta-data related to the weights of the model and its configuration. These can be
            informative attributes (for example the number of parameters/flops, recipe link/methods used in training
            etc), configuration parameters (for example the `num_classes`) needed to construct the model or important
            meta-data (for example the `classes` of a classification model) needed to use the model.
    """

    url: str
    transforms: Callable
    meta: Dict[str, Any]

    def __eq__(self, other: Any) -> bool:
        # We need this custom implementation for correct deep-copy and deserialization behavior.
        # TL;DR: After the definition of an enum, creating a new instance, i.e. by deep-copying or deserializing it,
        # involves an equality check against the defined members. Unfortunately, the `transforms` attribute is often
        # defined with `functools.partial` and `fn = partial(...); assert deepcopy(fn) != fn`. Without custom handling
        # for it, the check against the defined members would fail and effectively prevent the weights from being
        # deep-copied or deserialized.
        # See https://github.com/pytorch/vision/pull/7107 for details.
        if not isinstance(other, Weights):
            return NotImplemented

        if self.url != other.url:
            return False

        if self.meta != other.meta:
            return False

        if isinstance(self.transforms, partial) and isinstance(other.transforms, partial):
            return (
                    self.transforms.func == other.transforms.func
                    and self.transforms.args == other.transforms.args
                    and self.transforms.keywords == other.transforms.keywords
            )
        else:
            return self.transforms == other.transforms


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
    ) -> None:
        super().__init__()
        if norm_layer is None:
            if self.layer_norm:
                norm_layer = nn.LayerNorm
            else:
                norm_layer = nn.BatchNorm1d

        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
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


class decoder_block(nn.Module):
    def __init__(self, in_dconv, out_dconv):
        super().__init__()
        self.dconv = nn.ConvTranspose1d(in_dconv, 8 * out_dconv, kernel_size=3 , output_padding=1)
        self.drop = nn.Dropout(0.5)
        self.conv1 = nn.Conv1d(8 * out_dconv, 8 * out_dconv, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(8 * out_dconv, 8 * out_dconv, 3, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, inputs):
        x = self.dconv(inputs)
        x = self.drop(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)

        return x


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
            type_slow_fast: str,
            layer_norm,
            bottleneck_type: List[int],
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = (1, 0),
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.layer_norm=layer_norm
        if norm_layer is None:
            if self.layer_norm:
                norm_layer = nn.LayerNorm
            else:
                norm_layer = nn.BatchNorm1d

        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        if bottleneck_type == 1:
            self.conv1 = conv1d_1(inplanes, width)
            self.bn1 = norm_layer(width)
            if type_slow_fast == "slow":
                self.conv2 = conv1d_3(width, width, stride, groups, dilation)
            if type_slow_fast == "fast":
                self.conv2 = conv1d_16(width, width, stride, groups, dilation)
            self.bn2 = norm_layer(width)
            self.conv3 = conv1d_1(width, planes * self.expansion)

        if bottleneck_type == 2:
            if type_slow_fast == "slow":
                self.conv1 = conv1d_1(inplanes, width)
                self.bn1 = norm_layer(width)
                self.conv2 = conv1d_3(width, width, stride, groups, dilation)
            if type_slow_fast == "fast":
                self.conv1 = conv1d_1(inplanes, width)
                self.bn1 = norm_layer(width)
                self.conv2 = conv1d_16(width, width, stride, groups, dilation)
            self.bn2 = norm_layer(width)
            self.conv3 = conv1d_1(width, planes * self.expansion)

        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)

        if self.layer_norm:
            out = out.permute(0,2,1)
        out = self.bn1(out)
        if self.layer_norm:
            out = out.permute(0,2,1)

        out = self.relu(out)
        out = self.conv2(out)

        if self.layer_norm:
            out = out.permute(0,2,1)
        out = self.bn2(out)
        if self.layer_norm:
            out = out.permute(0,2,1)


        out = self.relu(out)
        out = self.conv3(out)

        if self.layer_norm:
            out = out.permute(0,2,1)
        out = self.bn3(out)
        if self.layer_norm:
            out = out.permute(0,2,1)

        if self.downsample is not None:
            identity = self.downsample[0](x)
            if self.layer_norm:
                identity = identity.permute(0,2,1)
            identity = self.downsample[1](identity)
            if self.layer_norm:
                identity = identity.permute(0,2,1)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            type_slow_fast: str,
            layer_norm,
            num_channels: int,
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            if layer_norm:
                norm_layer = nn.LayerNorm
            else:
                norm_layer = nn.BatchNorm1d

        self._norm_layer = norm_layer

        if type_slow_fast == "slow":
            self.inplanes = 64
        if type_slow_fast == "fast":
            self.inplanes = 8

        bottleneck_type = [1, 1, 2, 2]

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
        if type_slow_fast == "slow":
            self.maxpool = nn.MaxPool1d(kernel_size=1, stride=2)
            self.conv1 = nn.Conv1d(num_channels, self.inplanes, kernel_size=32, stride=1, padding='same', bias=False)
        if type_slow_fast == "fast":
            self.maxpool = nn.MaxPool1d(kernel_size=1, stride=1)
            self.conv1 = nn.Conv1d(num_channels, self.inplanes, kernel_size=8, stride=1, padding='same', bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        if type_slow_fast == "slow":
            self.layer1 = self._make_layer(block, 32, layers[0], stride=2, type_slow_fast=type_slow_fast,
                                           bottleneck_type=bottleneck_type[0], layer_norm=layer_norm)
            self.layer2 = self._make_layer(block, 64, layers[1], stride=2, dilate=replace_stride_with_dilation[0],
                                           type_slow_fast=type_slow_fast, bottleneck_type=bottleneck_type[1], layer_norm=layer_norm)
            self.layer3 = self._make_layer(block, 128, layers[2], stride=2, dilate=replace_stride_with_dilation[1],
                                           type_slow_fast=type_slow_fast, bottleneck_type=bottleneck_type[2], layer_norm=layer_norm)
            self.layer4 = self._make_layer(block, 256, layers[3], stride=2, dilate=replace_stride_with_dilation[2],
                                           type_slow_fast=type_slow_fast, bottleneck_type=bottleneck_type[3], layer_norm=layer_norm)

        if type_slow_fast == "fast":
            self.layer1 = self._make_layer(block, 4, layers[0], stride=2, type_slow_fast=type_slow_fast,
                                           bottleneck_type=bottleneck_type[0], layer_norm=layer_norm)
            self.layer2 = self._make_layer(block, 8, layers[1], stride=2, dilate=replace_stride_with_dilation[0],
                                           type_slow_fast=type_slow_fast, bottleneck_type=bottleneck_type[1], layer_norm=layer_norm)
            self.layer3 = self._make_layer(block, 16, layers[2], stride=2, dilate=replace_stride_with_dilation[1],
                                           type_slow_fast=type_slow_fast, bottleneck_type=bottleneck_type[2], layer_norm=layer_norm)
            self.layer4 = self._make_layer(block, 32, layers[3], stride=2, dilate=replace_stride_with_dilation[2],
                                           type_slow_fast=type_slow_fast, bottleneck_type=bottleneck_type[3], layer_norm=layer_norm)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        if layer_norm:
            self.normalize = nn.LayerNorm
        else:
            self.normalize = nn.BatchNorm1d

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (self.normalize, nn.GroupNorm)):
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
            type_slow_fast: str,
            layer_norm,
            bottleneck_type: int,
            stride: int = 1,
            dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        self.layer_norm = layer_norm
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1d_1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, type_slow_fast, layer_norm, bottleneck_type, stride, downsample, self.groups,
                self.base_width,
                previous_dilation, norm_layer
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
                    type_slow_fast=type_slow_fast,
                    bottleneck_type=bottleneck_type,
                    layer_norm=layer_norm,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tuple:
        # See note [TorchScript super()]
        x = self.maxpool(x)
        x = self.conv1(x)
        if self.layer_norm:
           x = x.permute(0, 2, 1)
        x = self.bn1(x)
        if self.layer_norm:
            x = x.permute(0, 2, 1)
        x = self.relu(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = torch.flatten(x, 1)

        return x, x1, x2, x3, x4

    def forward(self, x: Tensor) -> Tuple:
        return self._forward_impl(x)


def _resnet(
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        weights: Optional[WeightsEnum],
        progress: bool,
        type_slow_fast: str,  # Slow or fast
        layer_norm,
        num_channels: int,
        **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, type_slow_fast, layer_norm, num_channels, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


class slowfast_raw(nn.Module):
    def __init__(self, dropout, num_channels, layer_norm, stochastic_depth, add_FC_BN, remove_BN, add_FC, replaceBN_LN, add_FC_LN, add_2FC, input_size_seconds, slowfast_output, unet_output):
        super(slowfast_raw, self).__init__()
        self.dropout = dropout
        self.num_channels = num_channels
        self.layer_norm = layer_norm
        self.stochastic_depth = stochastic_depth
        self.add_FC_BN = add_FC_BN
        self.add_FC_LN = add_FC_LN
        self.add_FC = add_FC
        self.add_2FC = add_2FC
        self.remove_BN = remove_BN
        self.replaceBN_LN = replaceBN_LN
        self.slowfast_output = slowfast_output
        self.unet_output = unet_output
        self.input_size_seconds = input_size_seconds

        self.branch_slow = _resnet(Bottleneck, layers=[3, 3, 3, 3], weights=None, progress=False, type_slow_fast="slow", num_channels=self.num_channels, layer_norm=self.layer_norm)
        self.branch_fast = _resnet(Bottleneck, layers=[3, 3, 3, 3], weights=None, progress=False, type_slow_fast="fast", num_channels=self.num_channels, layer_norm=self.layer_norm)

        self.adapt_slow_1 = nn.Conv1d(128, 16, 1)

        if self.layer_norm:
            self.adapt_slow_1_bn = nn.LayerNorm(16)
        else:
            self.adapt_slow_1_bn = nn.BatchNorm1d(16)

        self.adapt_slow_relu_1 = nn.ReLU()
        self.adapt_slow_upsample_1 = nn.Upsample(scale_factor=2)

        self.adapt_slow_2 = nn.Conv1d(256, 32, 1)

        if self.layer_norm:
            self.adapt_slow_2_bn = nn.LayerNorm(32)
        else:
            self.adapt_slow_2_bn = nn.BatchNorm1d(32)

        self.adapt_slow_relu_2 = nn.ReLU()
        self.adapt_slow_upsample_2 = nn.Upsample(scale_factor=2)

        self.adapt_slow_3 = nn.Conv1d(512, 64, 1)

        if self.layer_norm:
            self.adapt_slow_3_bn = nn.LayerNorm(64)
        else:
            self.adapt_slow_3_bn = nn.BatchNorm1d(64)

        self.adapt_slow_relu_3 = nn.ReLU()
        self.adapt_slow_upsample_3 = nn.Upsample(scale_factor=2)

        self.adapt_slow_4 = nn.Conv1d(1024, 128, 1)

        if self.layer_norm:
            self.adapt_slow_4_bn = nn.LayerNorm(128)
        else:
            self.adapt_slow_4_bn = nn.BatchNorm1d(128)

        self.adapt_slow_relu_4 = nn.ReLU()
        self.adapt_slow_upsample_4 = nn.Upsample(scale_factor=2)

        if self.input_size_seconds==4:
            self.decoder_block1_dconv = nn.ConvTranspose1d(1152, 256, kernel_size=27, stride=8,
                                                           output_padding=5, groups=64)
        elif self.input_size_seconds==3:
            self.decoder_block1_dconv = nn.ConvTranspose1d(1152, 256, kernel_size=19, stride=8,
                                                           output_padding=5, groups=64)
        elif self.input_size_seconds==3.25:
            self.decoder_block1_dconv = nn.ConvTranspose1d(1152, 256, kernel_size=21, stride=8,
                                                           output_padding=5, groups=64)
        elif self.input_size_seconds==2:
            self.decoder_block1_dconv = nn.ConvTranspose1d(1152, 256, kernel_size=11, stride=8,
                                                           output_padding=5, groups=64)
        elif self.input_size_seconds==1:
            self.decoder_block1_dconv = nn.ConvTranspose1d(1152, 256, kernel_size=5, stride=8,
                                                           output_padding=3, groups=64)

        if self.layer_norm:
            self.decoder_block1_dconv_bn = nn.LayerNorm(256)
        else:
            self.decoder_block1_dconv_bn = nn.BatchNorm1d(256)

        if stochastic_depth > 0:
            self.drop1 = DropPath(stochastic_depth)
            self.drop1_layernorm = nn.LayerNorm(256)

        self.decoder_block1_drop = nn.Dropout(self.dropout)

        if self.stochastic_depth > 0:
            self.decoder_block1_conv1 = nn.Conv1d(256, 256, 3, padding=1)
        else:
            self.decoder_block1_conv1 = nn.Conv1d(512, 256, 3, padding=1)

        if self.layer_norm:
            self.decoder_block1_conv1_bn = nn.LayerNorm(256)
        else:
            self.decoder_block1_conv1_bn = nn.BatchNorm1d(256)

        self.decoder_block1_relu1 = nn.ReLU()


        self.decoder_block1_conv2 = nn.Conv1d(256, 256, 3, padding=1)

        if self.layer_norm:
            self.decoder_block1_conv2_bn = nn.LayerNorm(256)
        else:
            self.decoder_block1_conv2_bn = nn.BatchNorm1d(256)

        self.decoder_block1_relu2 = nn.ReLU()

        self.decoder_block2_dconv = nn.ConvTranspose1d(256, 128, kernel_size=6, stride=2, output_padding=0,
                                                       padding=2, groups=32)

        if self.layer_norm:
            self.decoder_block2_dconv_bn = nn.LayerNorm(128)
        else:
            self.decoder_block2_dconv_bn = nn.BatchNorm1d(128)

        if stochastic_depth > 0:
            self.drop2 = DropPath(stochastic_depth)
            self.drop2_layernorm = nn.LayerNorm(128)

        self.decoder_block2_drop = nn.Dropout(self.dropout)

        if self.stochastic_depth > 0:
            self.decoder_block2_conv1 = nn.Conv1d(128, 128, 3, padding=1)
        else:
            self.decoder_block2_conv1 = nn.Conv1d(256, 128, 3, padding=1)

        if self.layer_norm:
            self.decoder_block2_conv1_bn = nn.LayerNorm(128)
        else:
            self.decoder_block2_conv1_bn = nn.BatchNorm1d(128)

        self.decoder_block2_relu1 = nn.ReLU()
        self.decoder_block2_conv2 = nn.Conv1d(128, 128, 3, padding=1)

        if self.layer_norm:
            self.decoder_block2_conv2_bn = nn.LayerNorm(128)
        else:
            self.decoder_block2_conv2_bn = nn.BatchNorm1d(128)

        self.decoder_block2_relu2 = nn.ReLU()

        self.decoder_block3_dconv = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, output_padding=1,
                                                       padding=1, groups=16)

        if self.layer_norm:
            self.decoder_block3_dconv_bn = nn.LayerNorm(64)
        else:
            self.decoder_block3_dconv_bn = nn.BatchNorm1d(64)

        if stochastic_depth > 0:
            self.drop3 = DropPath(stochastic_depth)
            self.drop3_layernorm = nn.LayerNorm(64)

        self.decoder_block3_drop = nn.Dropout(self.dropout)

        if self.stochastic_depth > 0:
            self.decoder_block3_conv1 = nn.Conv1d(64, 64, 3, padding=1)
        else:
            self.decoder_block3_conv1 = nn.Conv1d(128, 64, 3, padding=1)

        if self.layer_norm:
            self.decoder_block3_conv1_bn = nn.LayerNorm(64)
        else:
            self.decoder_block3_conv1_bn = nn.BatchNorm1d(64)

        self.decoder_block3_relu1 = nn.ReLU()
        self.decoder_block3_conv2 = nn.Conv1d(64, 64, 3, padding=1)

        if self.layer_norm:
            self.decoder_block3_conv2_bn = nn.LayerNorm(64)
        else:
            self.decoder_block3_conv2_bn = nn.BatchNorm1d(64)

        self.decoder_block3_relu2 = nn.ReLU()

        self.decoder_block4_dconv = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, output_padding=1,
                                                       padding=1, groups=8)

        if self.layer_norm:
            self.decoder_block4_dconv_bn = nn.LayerNorm(32)
        else:
            self.decoder_block4_dconv_bn = nn.BatchNorm1d(32)

        if stochastic_depth > 0:
            self.drop4 = DropPath(stochastic_depth)
            self.drop4_layernorm = nn.LayerNorm(32)

        self.decoder_block4_drop = nn.Dropout(self.dropout)

        if self.stochastic_depth > 0:
            self.decoder_block4_conv1 = nn.Conv1d(32, 32, 3, padding=1)
        else:
            self.decoder_block4_conv1 = nn.Conv1d(64, 32, 3, padding=1)


        if self.layer_norm:
            self.decoder_block4_conv1_bn = nn.LayerNorm(32)
        else:
            self.decoder_block4_conv1_bn = nn.BatchNorm1d(32)

        self.decoder_block4_relu1 = nn.ReLU()
        self.decoder_block4_conv2 = nn.Conv1d(32, 32, 3, padding=1)

        if self.layer_norm:
            self.decoder_block4_conv2_bn = nn.LayerNorm(32)
        else:
            self.decoder_block4_conv2_bn = nn.BatchNorm1d(32)

        self.decoder_block4_relu2 = nn.ReLU()

        self.decoder_block5_dconv = nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, output_padding=1,
                                                       padding=1, groups=4)

        if self.layer_norm:
            self.decoder_block5_dconv_bn = nn.LayerNorm(16)
        else:
            self.decoder_block5_dconv_bn = nn.BatchNorm1d(16)

        self.decoder_block5_drop = nn.Dropout(self.dropout)
        self.decoder_block5_conv1 = nn.Conv1d(16, 8, 3, padding=1)

        if self.layer_norm:
            self.decoder_block5_conv1_bn = nn.LayerNorm(8)
        else:
            self.decoder_block5_conv1_bn = nn.BatchNorm1d(8)

        self.decoder_block5_relu1 = nn.ReLU()
        self.decoder_block5_conv2 = nn.Conv1d(8, self.num_channels, 3, padding=1)

        if self.replaceBN_LN:
            self.decoder_block5_conv2_bn = nn.LayerNorm(self.num_channels)
        else:
            if not remove_BN:
                if self.layer_norm:
                    self.decoder_block5_conv2_bn = nn.LayerNorm(self.num_channels)
                else:
                    self.decoder_block5_conv2_bn = nn.BatchNorm1d(self.num_channels)

        self.decoder_block5_relu2 = nn.ReLU()

        if self.add_FC_BN:
            self.linear1 = nn.Linear(512, 512)
            self.last_bn = nn.BatchNorm1d(self.num_channels)

        if self.add_FC and self.input_size_seconds==4:
            self.linear1 = nn.Linear(512, 512)
        elif self.add_FC and self.input_size_seconds==3:
            self.linear1 = nn.Linear(384, 384)
        elif self.add_FC and self.input_size_seconds==3.25:
            self.linear1 = nn.Linear(416, 416)
        elif self.add_FC and self.input_size_seconds==2:
            self.linear1 = nn.Linear(256, 256)
        elif self.add_FC and self.input_size_seconds==1:
            self.linear1 = nn.Linear(128, 128)

        if self.add_FC_LN:
            self.linear1 = nn.Linear(512, 512)
            self.last_LN = nn.LayerNorm(self.num_channels)
        if self.add_2FC:
            self.linear1 = nn.Linear(512, 512)
            self.linear2 = nn.Linear(512, 512)

    def forward(self, x: Tensor):
        out_slow, x1_slow, x2_slow, x3_slow, x4_slow = self.branch_slow(x)
        out_fast, x1_fast, x2_fast, x3_fast, x4_fast = self.branch_fast(x)
        out = torch.cat((out_slow, out_fast), dim=1)
        out = out.unsqueeze(2)

        x1_slow = self.adapt_slow_1(x1_slow)

        if self.layer_norm:
            x1_slow = x1_slow.permute(0,2,1)
        x1_slow = self.adapt_slow_1_bn(x1_slow)
        if self.layer_norm:
            x1_slow = x1_slow.permute(0,2,1)

        x1_slow = self.adapt_slow_relu_1(x1_slow)
        x1_slow = self.adapt_slow_upsample_1(x1_slow)
        x2_slow = self.adapt_slow_2(x2_slow)

        if self.layer_norm:
            x2_slow = x2_slow.permute(0,2,1)
        x2_slow = self.adapt_slow_2_bn(x2_slow)
        if self.layer_norm:
            x2_slow = x2_slow.permute(0,2,1)

        x2_slow = self.adapt_slow_relu_2(x2_slow)
        x2_slow = self.adapt_slow_upsample_2(x2_slow)

        x3_slow = self.adapt_slow_3(x3_slow)

        if self.layer_norm:
            x3_slow = x3_slow.permute(0,2,1)
        x3_slow = self.adapt_slow_3_bn(x3_slow)
        if self.layer_norm:
            x3_slow = x3_slow.permute(0,2,1)

        x3_slow = self.adapt_slow_relu_3(x3_slow)
        x3_slow = self.adapt_slow_upsample_3(x3_slow)

        x4_slow = self.adapt_slow_4(x4_slow)

        if self.layer_norm:
            x4_slow = x4_slow.permute(0,2,1)
        x4_slow = self.adapt_slow_4_bn(x4_slow)
        if self.layer_norm:
            x4_slow = x4_slow.permute(0,2,1)

        x4_slow = self.adapt_slow_relu_4(x4_slow)
        x4_slow = self.adapt_slow_upsample_4(x4_slow)

        dconv1 = self.decoder_block1_dconv(out)

        if self.layer_norm:
            dconv1 = dconv1.permute(0,2,1)
        dconv1 = self.decoder_block1_dconv_bn(dconv1)
        if self.layer_norm:
            dconv1 = dconv1.permute(0,2,1)

        if self.stochastic_depth > 0:
            concat_block1 = torch.add(self.drop1(dconv1), torch.cat((x4_slow, x4_fast), dim=1))
            concat_block1 = concat_block1.permute(0,2,1)
            concat_block1 = self.drop1_layernorm(concat_block1)
            concat_block1 = concat_block1.permute(0,2,1)
        else:
            concat_block1 = torch.cat((torch.cat((x4_slow, x4_fast), dim=1), dconv1), dim=1)

        concat_block1 = self.decoder_block1_drop(concat_block1)
        concat_block1 = self.decoder_block1_conv1(concat_block1)

        if self.layer_norm:
            concat_block1 = concat_block1.permute(0,2,1)
        concat_block1 = self.decoder_block1_conv1_bn(concat_block1)
        if self.layer_norm:
            concat_block1 = concat_block1.permute(0,2,1)

        concat_block1 = self.decoder_block1_relu1(concat_block1)
        concat_block1 = self.decoder_block1_conv2(concat_block1)

        if self.layer_norm:
            concat_block1 = concat_block1.permute(0,2,1)
        concat_block1 = self.decoder_block1_conv2_bn(concat_block1)
        if self.layer_norm:
            concat_block1 = concat_block1.permute(0,2,1)

        concat_block1 = self.decoder_block1_relu2(concat_block1)

        dconv2 = self.decoder_block2_dconv(concat_block1)

        if self.layer_norm:
            dconv2 = dconv2.permute(0,2,1)
        dconv2 = self.decoder_block2_dconv_bn(dconv2)
        if self.layer_norm:
            dconv2 = dconv2.permute(0,2,1)


        if self.stochastic_depth > 0:
            concat_block2 = torch.add(self.drop2(dconv2), torch.cat((x3_slow, x3_fast), dim=1))
            concat_block2 = concat_block2.permute(0,2,1)
            concat_block2 = self.drop2_layernorm(concat_block2)
            concat_block2 = concat_block2.permute(0,2,1)
        else:
            concat_block2 = torch.cat((torch.cat((x3_slow, x3_fast), dim=1), dconv2), dim=1)

        concat_block2 = self.decoder_block2_drop(concat_block2)
        concat_block2 = self.decoder_block2_conv1(concat_block2)

        if self.layer_norm:
            concat_block2 = concat_block2.permute(0,2,1)
        concat_block2 = self.decoder_block2_conv1_bn(concat_block2)
        if self.layer_norm:
            concat_block2 = concat_block2.permute(0,2,1)

        concat_block2 = self.decoder_block2_relu1(concat_block2)
        concat_block2 = self.decoder_block2_conv2(concat_block2)

        if self.layer_norm:
            concat_block2 = concat_block2.permute(0,2,1)
        concat_block2 = self.decoder_block2_conv2_bn(concat_block2)
        if self.layer_norm:
            concat_block2 = concat_block2.permute(0,2,1)

        concat_block2 = self.decoder_block2_relu2(concat_block2)

        dconv3 = self.decoder_block3_dconv(concat_block2)

        if self.layer_norm:
            dconv3 = dconv3.permute(0,2,1)
        dconv3 = self.decoder_block3_dconv_bn(dconv3)
        if self.layer_norm:
            dconv3 = dconv3.permute(0,2,1)

        if self.stochastic_depth > 0:
            concat_block3 = torch.add(self.drop3(dconv3), torch.cat((x2_slow, x2_fast), dim=1))
            concat_block3 = concat_block3.permute(0,2,1)
            concat_block3 = self.drop3_layernorm(concat_block3)
            concat_block3 = concat_block3.permute(0,2,1)
        else:
            concat_block3 = torch.cat((torch.cat((x2_slow, x2_fast), dim=1), dconv3), dim=1)

        concat_block3 = self.decoder_block3_drop(concat_block3)
        concat_block3 = self.decoder_block3_conv1(concat_block3)
        concat_block3 = self.decoder_block3_conv1_bn(concat_block3)
        concat_block3 = self.decoder_block3_relu1(concat_block3)
        concat_block3 = self.decoder_block3_conv2(concat_block3)

        if self.layer_norm:
            concat_block3 = concat_block3.permute(0,2,1)
        concat_block3 = self.decoder_block3_conv2_bn(concat_block3)
        if self.layer_norm:
            concat_block3 = concat_block3.permute(0,2,1)

        concat_block3 = self.decoder_block3_relu2(concat_block3)

        dconv4 = self.decoder_block4_dconv(concat_block3)

        if self.layer_norm:
            dconv4 = dconv4.permute(0,2,1)
        dconv4 = self.decoder_block4_dconv_bn(dconv4)
        if self.layer_norm:
            dconv4 = dconv4.permute(0,2,1)

        if self.stochastic_depth > 0:
            concat_block4 = torch.add(self.drop4(dconv4), torch.cat((x1_slow, x1_fast), dim=1))
            concat_block4 = concat_block4.permute(0,2,1)
            concat_block4 = self.drop4_layernorm(concat_block4)
            concat_block4 = concat_block4.permute(0,2,1)
        else:
            concat_block4 = torch.cat((torch.cat((x1_slow, x1_fast), dim=1), dconv4), dim=1)

        concat_block4 = self.decoder_block4_drop(concat_block4)
        concat_block4 = self.decoder_block4_conv1(concat_block4)

        if self.layer_norm:
            concat_block4 = concat_block4.permute(0,2,1)
        concat_block4 = self.decoder_block4_conv1_bn(concat_block4)
        if self.layer_norm:
            concat_block4 = concat_block4.permute(0,2,1)

        concat_block4 = self.decoder_block4_relu1(concat_block4)
        concat_block4 = self.decoder_block4_conv2(concat_block4)

        if self.layer_norm:
            concat_block4 = concat_block4.permute(0,2,1)
        concat_block4 = self.decoder_block4_conv2_bn(concat_block4)
        if self.layer_norm:
            concat_block4 = concat_block4.permute(0,2,1)

        concat_block4 = self.decoder_block4_relu2(concat_block4)

        dconv5 = self.decoder_block5_dconv(concat_block4)

        if self.layer_norm:
            dconv5 = dconv5.permute(0, 2, 1)
        dconv5 = self.decoder_block5_dconv_bn(dconv5)
        if self.layer_norm:
            dconv5 = dconv5.permute(0, 2, 1)

        concat_block5 = self.decoder_block5_drop(dconv5)
        concat_block5 = self.decoder_block5_conv1(concat_block5)

        if self.layer_norm:
            concat_block5 = concat_block5.permute(0, 2, 1)
        concat_block5 = self.decoder_block5_conv1_bn(concat_block5)
        if self.layer_norm:
            concat_block5 = concat_block5.permute(0, 2, 1)

        concat_block5 = self.decoder_block5_relu1(concat_block5)
        concat_block5 = self.decoder_block5_conv2(concat_block5)

        if self.layer_norm:
            concat_block5 = concat_block5.permute(0, 2, 1)

        if self.replaceBN_LN:
            concat_block5 = concat_block5.permute(0, 2, 1)
            concat_block5 = self.decoder_block5_conv2_bn(concat_block5)
            concat_block5 = concat_block5.permute(0, 2, 1)
        else:
            if not self.remove_BN:
                concat_block5 = self.decoder_block5_conv2_bn(concat_block5)

        if self.layer_norm:
            concat_block5 = concat_block5.permute(0, 2, 1)

        if self.add_FC_BN:
            concat_block5 = self.linear1(concat_block5)
            concat_block5 = self.last_bn(concat_block5)
        if self.add_FC:
            concat_block5 = self.linear1(concat_block5)

        if self.add_2FC:
            concat_block5 = self.linear1(concat_block5)
            concat_block5 = self.linear2(concat_block5)

        if self.add_FC_LN:
            concat_block5 = self.linear1(concat_block5)
            concat_block5 = concat_block5.permute(0, 2, 1)
            concat_block5 = self.last_LN(concat_block5)
            concat_block5 = concat_block5.permute(0, 2, 1)

        if (self.unet_output and self.slowfast_output):
            return out, concat_block5
        elif self.unet_output:
            return concat_block5
        elif self.slowfast_output:
            return out
        return concat_block5

class slowfast_raw_sum(nn.Module):
    def __init__(self, dropout, num_channels, layer_norm, stochastic_depth, add_FC_BN, remove_BN, add_FC, replaceBN_LN, add_FC_LN, add_2FC):
        super(slowfast_raw_sum, self).__init__()
        self.dropout = dropout
        self.num_channels = num_channels
        self.layer_norm = layer_norm
        self.stochastic_depth = stochastic_depth
        self.add_FC_BN = add_FC_BN
        self.add_FC_LN = add_FC_LN
        self.add_FC = add_FC
        self.add_2FC = add_2FC
        self.remove_BN = remove_BN
        self.replaceBN_LN = replaceBN_LN

        self.branch_slow = _resnet(Bottleneck, layers=[3, 3, 3, 3], weights=None, progress=False, type_slow_fast="slow", num_channels=self.num_channels, layer_norm=self.layer_norm)
        self.branch_fast = _resnet(Bottleneck, layers=[3, 3, 3, 3], weights=None, progress=False, type_slow_fast="fast", num_channels=self.num_channels, layer_norm=self.layer_norm)

        self.adapt_slow_1 = nn.Conv1d(128, 16, 1)

        if self.layer_norm:
            self.adapt_slow_1_bn = nn.LayerNorm(16)
        else:
            self.adapt_slow_1_bn = nn.BatchNorm1d(16)

        self.adapt_slow_relu_1 = nn.ReLU()
        self.adapt_slow_upsample_1 = nn.Upsample(scale_factor=2)

        self.adapt_slow_2 = nn.Conv1d(256, 32, 1)

        if self.layer_norm:
            self.adapt_slow_2_bn = nn.LayerNorm(32)
        else:
            self.adapt_slow_2_bn = nn.BatchNorm1d(32)

        self.adapt_slow_relu_2 = nn.ReLU()
        self.adapt_slow_upsample_2 = nn.Upsample(scale_factor=2)

        self.adapt_slow_3 = nn.Conv1d(512, 64, 1)

        if self.layer_norm:
            self.adapt_slow_3_bn = nn.LayerNorm(64)
        else:
            self.adapt_slow_3_bn = nn.BatchNorm1d(64)

        self.adapt_slow_relu_3 = nn.ReLU()
        self.adapt_slow_upsample_3 = nn.Upsample(scale_factor=2)

        self.adapt_slow_4 = nn.Conv1d(1024, 128, 1)

        if self.layer_norm:
            self.adapt_slow_4_bn = nn.LayerNorm(128)
        else:
            self.adapt_slow_4_bn = nn.BatchNorm1d(128)

        self.adapt_slow_relu_4 = nn.ReLU()
        self.adapt_slow_upsample_4 = nn.Upsample(scale_factor=2)


        self.decoder_block1_dconv = nn.ConvTranspose1d(1152, 256, kernel_size=27, stride=8,
                                                       output_padding=5, groups=64)

        if self.layer_norm:
            self.decoder_block1_dconv_bn = nn.LayerNorm(256)
        else:
            self.decoder_block1_dconv_bn = nn.BatchNorm1d(256)

        if stochastic_depth > 0:
            self.drop1 = DropPath(stochastic_depth)
            self.drop1_layernorm = nn.LayerNorm(256)

        self.decoder_block1_drop = nn.Dropout(self.dropout)

        if self.stochastic_depth > 0:
            self.decoder_block1_conv1 = nn.Conv1d(256, 256, 3, padding=1)
        else:
            self.decoder_block1_conv1 = nn.Conv1d(512, 256, 3, padding=1)

        if self.layer_norm:
            self.decoder_block1_conv1_bn = nn.LayerNorm(256)
        else:
            self.decoder_block1_conv1_bn = nn.BatchNorm1d(256)

        self.decoder_block1_relu1 = nn.ReLU()


        self.decoder_block1_conv2 = nn.Conv1d(256, 256, 3, padding=1)

        if self.layer_norm:
            self.decoder_block1_conv2_bn = nn.LayerNorm(256)
        else:
            self.decoder_block1_conv2_bn = nn.BatchNorm1d(256)

        self.decoder_block1_relu2 = nn.ReLU()

        self.decoder_block2_dconv = nn.ConvTranspose1d(256, 128, kernel_size=6, stride=2, output_padding=0,
                                                       padding=2, groups=32)

        if self.layer_norm:
            self.decoder_block2_dconv_bn = nn.LayerNorm(128)
        else:
            self.decoder_block2_dconv_bn = nn.BatchNorm1d(128)

        if stochastic_depth > 0:
            self.drop2 = DropPath(stochastic_depth)
            self.drop2_layernorm = nn.LayerNorm(128)

        self.decoder_block2_drop = nn.Dropout(self.dropout)

        if self.stochastic_depth > 0:
            self.decoder_block2_conv1 = nn.Conv1d(128, 128, 3, padding=1)
        else:
            self.decoder_block2_conv1 = nn.Conv1d(256, 128, 3, padding=1)

        if self.layer_norm:
            self.decoder_block2_conv1_bn = nn.LayerNorm(128)
        else:
            self.decoder_block2_conv1_bn = nn.BatchNorm1d(128)

        self.decoder_block2_relu1 = nn.ReLU()
        self.decoder_block2_conv2 = nn.Conv1d(128, 128, 3, padding=1)

        if self.layer_norm:
            self.decoder_block2_conv2_bn = nn.LayerNorm(128)
        else:
            self.decoder_block2_conv2_bn = nn.BatchNorm1d(128)

        self.decoder_block2_relu2 = nn.ReLU()

        self.decoder_block3_dconv = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, output_padding=1,
                                                       padding=1, groups=16)

        if self.layer_norm:
            self.decoder_block3_dconv_bn = nn.LayerNorm(64)
        else:
            self.decoder_block3_dconv_bn = nn.BatchNorm1d(64)

        if stochastic_depth > 0:
            self.drop3 = DropPath(stochastic_depth)
            self.drop3_layernorm = nn.LayerNorm(64)

        self.decoder_block3_drop = nn.Dropout(self.dropout)

        if self.stochastic_depth > 0:
            self.decoder_block3_conv1 = nn.Conv1d(64, 64, 3, padding=1)
        else:
            self.decoder_block3_conv1 = nn.Conv1d(128, 64, 3, padding=1)

        if self.layer_norm:
            self.decoder_block3_conv1_bn = nn.LayerNorm(64)
        else:
            self.decoder_block3_conv1_bn = nn.BatchNorm1d(64)

        self.decoder_block3_relu1 = nn.ReLU()
        self.decoder_block3_conv2 = nn.Conv1d(64, 64, 3, padding=1)

        if self.layer_norm:
            self.decoder_block3_conv2_bn = nn.LayerNorm(64)
        else:
            self.decoder_block3_conv2_bn = nn.BatchNorm1d(64)

        self.decoder_block3_relu2 = nn.ReLU()

        self.decoder_block4_dconv = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, output_padding=1,
                                                       padding=1, groups=8)

        if self.layer_norm:
            self.decoder_block4_dconv_bn = nn.LayerNorm(32)
        else:
            self.decoder_block4_dconv_bn = nn.BatchNorm1d(32)

        if stochastic_depth > 0:
            self.drop4 = DropPath(stochastic_depth)
            self.drop4_layernorm = nn.LayerNorm(32)

        self.decoder_block4_drop = nn.Dropout(self.dropout)

        if self.stochastic_depth > 0:
            self.decoder_block4_conv1 = nn.Conv1d(32, 32, 3, padding=1)
        else:
            self.decoder_block4_conv1 = nn.Conv1d(64, 32, 3, padding=1)


        if self.layer_norm:
            self.decoder_block4_conv1_bn = nn.LayerNorm(32)
        else:
            self.decoder_block4_conv1_bn = nn.BatchNorm1d(32)

        self.decoder_block4_relu1 = nn.ReLU()
        self.decoder_block4_conv2 = nn.Conv1d(32, 32, 3, padding=1)

        if self.layer_norm:
            self.decoder_block4_conv2_bn = nn.LayerNorm(32)
        else:
            self.decoder_block4_conv2_bn = nn.BatchNorm1d(32)

        self.decoder_block4_relu2 = nn.ReLU()

        self.decoder_block5_dconv = nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, output_padding=1,
                                                       padding=1, groups=4)

        if self.layer_norm:
            self.decoder_block5_dconv_bn = nn.LayerNorm(16)
        else:
            self.decoder_block5_dconv_bn = nn.BatchNorm1d(16)

        self.decoder_block5_drop = nn.Dropout(self.dropout)
        self.decoder_block5_conv1 = nn.Conv1d(16, 8, 3, padding=1)

        if self.layer_norm:
            self.decoder_block5_conv1_bn = nn.LayerNorm(8)
        else:
            self.decoder_block5_conv1_bn = nn.BatchNorm1d(8)

        self.decoder_block5_relu1 = nn.ReLU()
        self.decoder_block5_conv2 = nn.Conv1d(8, self.num_channels, 3, padding=1)

        if self.replaceBN_LN:
            self.decoder_block5_conv2_bn = nn.LayerNorm(self.num_channels)
        else:
            if not remove_BN:
                if self.layer_norm:
                    self.decoder_block5_conv2_bn = nn.LayerNorm(self.num_channels)
                else:
                    self.decoder_block5_conv2_bn = nn.BatchNorm1d(self.num_channels)

        self.decoder_block5_relu2 = nn.ReLU()

        if self.add_FC_BN:
            self.linear1 = nn.Linear(512, 512)
            self.last_bn = nn.BatchNorm1d(self.num_channels)
        if self.add_FC:
            self.linear1 = nn.Linear(512, 512)
        if self.add_FC_LN:
            self.linear1 = nn.Linear(512, 512)
            self.last_LN = nn.LayerNorm(self.num_channels)
        if self.add_2FC:
            self.linear1 = nn.Linear(512, 512)
            self.linear2 = nn.Linear(512, 512)

    def forward(self, x: Tensor):
        out_slow, x1_slow, x2_slow, x3_slow, x4_slow = self.branch_slow(x)
        out_fast, x1_fast, x2_fast, x3_fast, x4_fast = self.branch_fast(x)
        out = torch.cat((out_slow, out_fast), dim=1)  # Output slow and fast final
        out = out.unsqueeze(2)

        output_required = x1_slow

        x1_slow = self.adapt_slow_1(x1_slow)

        if self.layer_norm:
            x1_slow = x1_slow.permute(0,2,1)
        x1_slow = self.adapt_slow_1_bn(x1_slow)
        if self.layer_norm:
            x1_slow = x1_slow.permute(0,2,1)

        x1_slow = self.adapt_slow_relu_1(x1_slow)
        x1_slow = self.adapt_slow_upsample_1(x1_slow)
        x2_slow = self.adapt_slow_2(x2_slow)

        if self.layer_norm:
            x2_slow = x2_slow.permute(0,2,1)
        x2_slow = self.adapt_slow_2_bn(x2_slow)
        if self.layer_norm:
            x2_slow = x2_slow.permute(0,2,1)

        x2_slow = self.adapt_slow_relu_2(x2_slow)
        x2_slow = self.adapt_slow_upsample_2(x2_slow)

        x3_slow = self.adapt_slow_3(x3_slow)

        if self.layer_norm:
            x3_slow = x3_slow.permute(0,2,1)
        x3_slow = self.adapt_slow_3_bn(x3_slow)
        if self.layer_norm:
            x3_slow = x3_slow.permute(0,2,1)

        x3_slow = self.adapt_slow_relu_3(x3_slow)
        x3_slow = self.adapt_slow_upsample_3(x3_slow)

        x4_slow = self.adapt_slow_4(x4_slow)

        if self.layer_norm:
            x4_slow = x4_slow.permute(0,2,1)
        x4_slow = self.adapt_slow_4_bn(x4_slow)
        if self.layer_norm:
            x4_slow = x4_slow.permute(0,2,1)

        x4_slow = self.adapt_slow_relu_4(x4_slow)
        x4_slow = self.adapt_slow_upsample_4(x4_slow)

        dconv1 = self.decoder_block1_dconv(out)

        if self.layer_norm:
            dconv1 = dconv1.permute(0,2,1)
        dconv1 = self.decoder_block1_dconv_bn(dconv1)
        if self.layer_norm:
            dconv1 = dconv1.permute(0,2,1)

        if self.stochastic_depth > 0:
            concat_block1 = torch.add(self.drop1(dconv1), torch.cat((x4_slow, x4_fast), dim=1))
            concat_block1 = concat_block1.permute(0,2,1)
            concat_block1 = self.drop1_layernorm(concat_block1)
            concat_block1 = concat_block1.permute(0,2,1)
        else:
            concat_block1 = torch.cat((torch.cat((x4_slow, x4_fast), dim=1), dconv1), dim=1)

        concat_block1 = self.decoder_block1_drop(concat_block1)
        concat_block1 = self.decoder_block1_conv1(concat_block1)

        if self.layer_norm:
            concat_block1 = concat_block1.permute(0,2,1)
        concat_block1 = self.decoder_block1_conv1_bn(concat_block1)
        if self.layer_norm:
            concat_block1 = concat_block1.permute(0,2,1)

        concat_block1 = self.decoder_block1_relu1(concat_block1)
        concat_block1 = self.decoder_block1_conv2(concat_block1)

        if self.layer_norm:
            concat_block1 = concat_block1.permute(0,2,1)
        concat_block1 = self.decoder_block1_conv2_bn(concat_block1)
        if self.layer_norm:
            concat_block1 = concat_block1.permute(0,2,1)

        concat_block1 = self.decoder_block1_relu2(concat_block1)

        dconv2 = self.decoder_block2_dconv(concat_block1)

        if self.layer_norm:
            dconv2 = dconv2.permute(0,2,1)
        dconv2 = self.decoder_block2_dconv_bn(dconv2)
        if self.layer_norm:
            dconv2 = dconv2.permute(0,2,1)


        if self.stochastic_depth > 0:
            concat_block2 = torch.add(self.drop2(dconv2), torch.cat((x3_slow, x3_fast), dim=1))
            concat_block2 = concat_block2.permute(0,2,1)
            concat_block2 = self.drop2_layernorm(concat_block2)
            concat_block2 = concat_block2.permute(0,2,1)
        else:
            concat_block2 = torch.cat((torch.cat((x3_slow, x3_fast), dim=1), dconv2), dim=1)

        concat_block2 = self.decoder_block2_drop(concat_block2)
        concat_block2 = self.decoder_block2_conv1(concat_block2)

        if self.layer_norm:
            concat_block2 = concat_block2.permute(0,2,1)
        concat_block2 = self.decoder_block2_conv1_bn(concat_block2)
        if self.layer_norm:
            concat_block2 = concat_block2.permute(0,2,1)

        concat_block2 = self.decoder_block2_relu1(concat_block2)
        concat_block2 = self.decoder_block2_conv2(concat_block2)

        if self.layer_norm:
            concat_block2 = concat_block2.permute(0,2,1)
        concat_block2 = self.decoder_block2_conv2_bn(concat_block2)
        if self.layer_norm:
            concat_block2 = concat_block2.permute(0,2,1)

        concat_block2 = self.decoder_block2_relu2(concat_block2)

        dconv3 = self.decoder_block3_dconv(concat_block2)

        if self.layer_norm:
            dconv3 = dconv3.permute(0,2,1)
        dconv3 = self.decoder_block3_dconv_bn(dconv3)
        if self.layer_norm:
            dconv3 = dconv3.permute(0,2,1)

        if self.stochastic_depth > 0:
            concat_block3 = torch.add(self.drop3(dconv3), torch.cat((x2_slow, x2_fast), dim=1))
            concat_block3 = concat_block3.permute(0,2,1)
            concat_block3 = self.drop3_layernorm(concat_block3)
            concat_block3 = concat_block3.permute(0,2,1)
        else:
            concat_block3 = torch.cat((torch.cat((x2_slow, x2_fast), dim=1), dconv3), dim=1)

        concat_block3 = self.decoder_block3_drop(concat_block3)
        concat_block3 = self.decoder_block3_conv1(concat_block3)
        concat_block3 = self.decoder_block3_conv1_bn(concat_block3)
        concat_block3 = self.decoder_block3_relu1(concat_block3)
        concat_block3 = self.decoder_block3_conv2(concat_block3)

        if self.layer_norm:
            concat_block3 = concat_block3.permute(0,2,1)
        concat_block3 = self.decoder_block3_conv2_bn(concat_block3)
        if self.layer_norm:
            concat_block3 = concat_block3.permute(0,2,1)

        concat_block3 = self.decoder_block3_relu2(concat_block3)

        dconv4 = self.decoder_block4_dconv(concat_block3)

        if self.layer_norm:
            dconv4 = dconv4.permute(0,2,1)
        dconv4 = self.decoder_block4_dconv_bn(dconv4)
        if self.layer_norm:
            dconv4 = dconv4.permute(0,2,1)

        if self.stochastic_depth > 0:
            concat_block4 = torch.add(self.drop4(dconv4), torch.cat((x1_slow, x1_fast), dim=1))
            concat_block4 = concat_block4.permute(0,2,1)
            concat_block4 = self.drop4_layernorm(concat_block4)
            concat_block4 = concat_block4.permute(0,2,1)
        else:
            concat_block4 = torch.cat((torch.cat((x1_slow, x1_fast), dim=1), dconv4), dim=1)

        concat_block4 = self.decoder_block4_drop(concat_block4)
        concat_block4 = self.decoder_block4_conv1(concat_block4)

        if self.layer_norm:
            concat_block4 = concat_block4.permute(0,2,1)
        concat_block4 = self.decoder_block4_conv1_bn(concat_block4)
        if self.layer_norm:
            concat_block4 = concat_block4.permute(0,2,1)

        concat_block4 = self.decoder_block4_relu1(concat_block4)
        concat_block4 = self.decoder_block4_conv2(concat_block4)

        if self.layer_norm:
            concat_block4 = concat_block4.permute(0,2,1)
        concat_block4 = self.decoder_block4_conv2_bn(concat_block4)
        if self.layer_norm:
            concat_block4 = concat_block4.permute(0,2,1)

        concat_block4 = self.decoder_block4_relu2(concat_block4)

        dconv5 = self.decoder_block5_dconv(concat_block4)

        if self.layer_norm:
            dconv5 = dconv5.permute(0, 2, 1)
        dconv5 = self.decoder_block5_dconv_bn(dconv5)
        if self.layer_norm:
            dconv5 = dconv5.permute(0, 2, 1)

        concat_block5 = self.decoder_block5_drop(dconv5)
        concat_block5 = self.decoder_block5_conv1(concat_block5)

        if self.layer_norm:
            concat_block5 = concat_block5.permute(0, 2, 1)
        concat_block5 = self.decoder_block5_conv1_bn(concat_block5)
        if self.layer_norm:
            concat_block5 = concat_block5.permute(0, 2, 1)

        concat_block5 = self.decoder_block5_relu1(concat_block5)
        concat_block5 = self.decoder_block5_conv2(concat_block5)

        if self.layer_norm:
            concat_block5 = concat_block5.permute(0, 2, 1)

        if self.replaceBN_LN:
            concat_block5 = concat_block5.permute(0, 2, 1)
            concat_block5 = self.decoder_block5_conv2_bn(concat_block5)
            concat_block5 = concat_block5.permute(0, 2, 1)
        else:
            if not self.remove_BN:
                concat_block5 = self.decoder_block5_conv2_bn(concat_block5)

        if self.layer_norm:
            concat_block5 = concat_block5.permute(0, 2, 1)

        if self.add_FC_BN:
            concat_block5 = self.linear1(concat_block5)
            concat_block5 = self.last_bn(concat_block5)
        if self.add_FC:
            concat_block5 = self.linear1(concat_block5)

        if self.add_2FC:
            concat_block5 = self.linear1(concat_block5)
            concat_block5 = self.linear2(concat_block5)

        if self.add_FC_LN:
            concat_block5 = self.linear1(concat_block5)
            concat_block5 = concat_block5.permute(0, 2, 1)
            concat_block5 = self.last_LN(concat_block5)
            concat_block5 = concat_block5.permute(0, 2, 1)

        return concat_block5, out_slow, output_required, x2_slow, x3_slow, x4_slow, out_fast, x1_fast, x2_fast, x3_fast, x4_fast
