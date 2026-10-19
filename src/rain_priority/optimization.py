import torch
from accelerate import Accelerator


def backward_rain_priority(
    accelerator: Accelerator,
    total_loss: torch.Tensor,
    rain_loss: torch.Tensor,
    gate_parameters: list[torch.nn.Parameter],
) -> None:
    if accelerator.num_processes != 1 or accelerator.scaler is not None:
        raise ValueError("Rain-priority gradient routing supports one device with FP32/BF16 only.")
    previous = [None if p.grad is None else p.grad.detach().clone() for p in gate_parameters]
    gradients = (
        torch.autograd.grad(
            rain_loss / accelerator.gradient_accumulation_steps,
            gate_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        if gate_parameters
        else []
    )
    accelerator.backward(total_loss)
    for parameter, before, current in zip(gate_parameters, previous, gradients, strict=True):
        if current is None:
            parameter.grad = before
        else:
            parameter.grad = current.detach() if before is None else before + current.detach()
