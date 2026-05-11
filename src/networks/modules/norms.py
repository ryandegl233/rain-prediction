from functools import partial
from inspect import Parameter, isclass, isfunction, signature
from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, scale_factor=1.0, eps: float = 1e-6):
        """
            Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim) * scale_factor)

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        return (self.weight * self._norm(x.float())).type_as(x)


class RMSNorm2d(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = torch.nn.parameter.Parameter(torch.ones(self.num_features))
            if bias:
                self.bias = torch.nn.parameter.Parameter(torch.zeros(self.num_features))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x / torch.sqrt(torch.square(x.float()).mean(dim=1, keepdim=True) + self.eps)).to(
            x.dtype
        )
        if self.elementwise_affine:
            x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x


class LayerNorm2d(torch.nn.LayerNorm):
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


def Normalize(
    in_channels,
    norm_type: str
    | Literal["gn", "bn2d", "ln2d", "rms_native", "rms_triton", "unit_vec_norm", "none"]
    | None = "gn",
    **norm_kwargs,
):
    if norm_type == "gn":
        return torch.nn.GroupNorm(
            num_channels=in_channels,
            eps=1e-6,
            affine=True,
            num_groups=norm_kwargs.get("num_groups", 32),
        )
    elif norm_type == "bn2d":
        cls = torch.nn.BatchNorm2d
    elif norm_type == "ln2d":
        cls = LayerNorm2d
    elif norm_type == "rms_native":
        cls = RMSNorm2d
    elif norm_type in (None, "none"):
        return torch.nn.Identity()
    else:
        raise ValueError(
            f"Unknown normalization type: {norm_type}. Supported types are: 'gn', 'bn2d', 'ln2d', 'rms_native', "
            "'rms_triton', None or 'none'."
        )

    return cls(in_channels, **extract_needed_kwargs(norm_kwargs, cls))


def extract_needed_kwargs(
    kwargs: dict, cls: Callable | type, include_default: bool = False
) -> dict:
    """
    Extracts the subset of `kwargs` that match the parameters of a given class's __init__ method or a function.

    If a parameter is not provided in `kwargs` but has a default value, the default is used.
    Missing required parameters will raise a ValueError.

    Args:
        kwargs (dict): A dictionary of keyword arguments to filter.
        cls (type or function): A class or function whose signature is used to extract the needed kwargs.

    Returns:
        dict: A dictionary containing only the relevant keyword arguments.

    Raises:
        AssertionError: If `cls` is a class without an __init__ method.
        ValueError: If a required argument is missing or an unsupported type is passed.
    """
    needed_kwargs = {}
    if isclass(cls):
        assert hasattr(cls, "__init__"), f"{cls} does not have an __init__ method."
        sig = signature(cls.__init__)
    elif isfunction(cls):
        sig = signature(cls)
    else:
        raise ValueError(f"Expected a class or function, got {type(cls)}.")

    for param in sig.parameters.values():
        if param.name == "self":
            continue
        if param.name in kwargs:
            needed_kwargs[param.name] = kwargs[param.name]
        elif include_default and param.default is not Parameter.empty:
            needed_kwargs[param.name] = param.default
        # else:
        #     raise ValueError(
        #         f"Missing required argument '{param.name}' for {cls.__name__}."
        #     )
    return needed_kwargs
