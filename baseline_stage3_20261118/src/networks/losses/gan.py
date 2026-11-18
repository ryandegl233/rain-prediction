import torch
import torch.nn.functional as F
from typing import Literal

GanLossType = Literal["ns", "rel_ns", "hinge"]


def _validate_loss_type(loss_type: str) -> GanLossType:
    normalized = loss_type.strip().lower()
    if normalized not in {"ns", "rel_ns", "hinge"}:
        raise ValueError(f"Unsupported GAN loss_type='{loss_type}'. Expected one of ['ns', 'rel_ns', 'hinge'].")
    return normalized  # type: ignore[return-value]


def gan_generator_loss(
    fake_logits: torch.Tensor,
    real_logits: torch.Tensor | None = None,
    *,
    loss_type: str = "ns",
    weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    resolved_type = _validate_loss_type(loss_type)

    if resolved_type == "ns":
        loss_raw = F.softplus(-fake_logits.float()).mean()
    elif resolved_type == "rel_ns":
        if real_logits is None:
            raise ValueError("gan_generator_loss(loss_type='rel_ns') requires real_logits.")
        relative_fake = fake_logits.float() - real_logits.float()
        loss_raw = F.softplus(-relative_fake).mean()
    else:
        loss_raw = -fake_logits.float().mean()

    loss = loss_raw * float(weight)
    logs = {
        "gan/g_loss": loss.detach(),
        "gan/g_loss_raw": loss_raw.detach(),
    }
    return loss, logs


def gan_discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    *,
    loss_type: str = "ns",
    weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    resolved_type = _validate_loss_type(loss_type)

    if resolved_type == "ns":
        loss_real = F.softplus(-real_logits.float()).mean()
        loss_fake = F.softplus(fake_logits.float()).mean()
    elif resolved_type == "rel_ns":
        relative_real = real_logits.float() - fake_logits.float()
        loss_real = F.softplus(-relative_real).mean()
        loss_fake = torch.zeros_like(loss_real)
    else:
        loss_real = F.relu(1.0 - real_logits.float()).mean()
        loss_fake = F.relu(1.0 + fake_logits.float()).mean()

    loss_raw = loss_real + loss_fake
    loss = loss_raw * float(weight)
    logs = {
        "gan/d_loss": loss.detach(),
        "gan/d_loss_raw": loss_raw.detach(),
        "gan/d_loss_real": loss_real.detach(),
        "gan/d_loss_fake": loss_fake.detach(),
    }
    return loss, logs


def _grad_penalty_from_logits(logits: torch.Tensor, inputs: torch.Tensor, name: str) -> torch.Tensor:
    if not inputs.requires_grad:
        raise ValueError(f"{name} requires input tensor with requires_grad=True.")
    if not logits.requires_grad:
        raise ValueError(f"{name} requires logits that keep autograd graph.")

    grads = torch.autograd.grad(
        outputs=logits.sum(),
        inputs=inputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    if grads is None:
        raise ValueError(f"{name} failed to compute gradients.")
    grad_norm_sq = grads.float().pow(2).flatten(start_dim=1).sum(dim=1)
    return grad_norm_sq.mean()


def r1_regularization(
    real_logits: torch.Tensor,
    real_input: torch.Tensor,
    *,
    weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if float(weight) <= 0.0:
        zero = real_logits.new_zeros(())
        return zero, {"gan/r1": zero.detach(), "gan/r1_penalty": zero.detach()}

    penalty = _grad_penalty_from_logits(real_logits, real_input, name="r1_regularization")
    loss = penalty * float(weight)
    logs = {
        "gan/r1": loss.detach(),
        "gan/r1_penalty": penalty.detach(),
    }
    return loss, logs


def r2_regularization(
    fake_logits: torch.Tensor,
    fake_input: torch.Tensor,
    *,
    weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if float(weight) <= 0.0:
        zero = fake_logits.new_zeros(())
        return zero, {"gan/r2": zero.detach(), "gan/r2_penalty": zero.detach()}

    penalty = _grad_penalty_from_logits(fake_logits, fake_input, name="r2_regularization")
    loss = penalty * float(weight)
    logs = {
        "gan/r2": loss.detach(),
        "gan/r2_penalty": penalty.detach(),
    }
    return loss, logs


def gan_critic_total_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    *,
    loss_type: str = "ns",
    d_weight: float = 1.0,
    real_input: torch.Tensor | None = None,
    fake_input: torch.Tensor | None = None,
    r1_weight: float = 0.0,
    r2_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    d_loss, d_logs = gan_discriminator_loss(
        real_logits=real_logits,
        fake_logits=fake_logits,
        loss_type=loss_type,
        weight=d_weight,
    )

    if float(r1_weight) > 0.0:
        if real_input is None:
            raise ValueError("gan_critic_total_loss requires real_input when r1_weight > 0.")
        r1_loss, r1_logs = r1_regularization(real_logits=real_logits, real_input=real_input, weight=r1_weight)
    else:
        zero = real_logits.new_zeros(())
        r1_loss, r1_logs = zero, {"gan/r1": zero.detach(), "gan/r1_penalty": zero.detach()}

    if float(r2_weight) > 0.0:
        if fake_input is None:
            raise ValueError("gan_critic_total_loss requires fake_input when r2_weight > 0.")
        r2_loss, r2_logs = r2_regularization(fake_logits=fake_logits, fake_input=fake_input, weight=r2_weight)
    else:
        zero = fake_logits.new_zeros(())
        r2_loss, r2_logs = zero, {"gan/r2": zero.detach(), "gan/r2_penalty": zero.detach()}

    total_loss = d_loss + r1_loss + r2_loss
    logs = {
        **d_logs,
        **r1_logs,
        **r2_logs,
        "gan/critic_total_loss": total_loss.detach(),
    }
    return total_loss, logs
