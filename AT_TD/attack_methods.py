import logging
import math
from typing import Any, List, Optional, Tuple

import tensorly as tl
import torch

import TR_func as tr


def objective_function_ATTR(X, G_list, E, inner_num_steps: int = 100, inner_tol: float = 1e-10):
    """Compute -||X - TR(G)||_F^2 for the attacked tensor X+E."""
    X_tube = X + E
    tr_cores = tr.tr_decompose2(
        X_tube,
        G_list,
        max_iter=inner_num_steps,
        tol=inner_tol,
        dis_num=2000,
        device="cuda",
    )
    TN_G = tl.tr_to_tensor(tr_cores)
    return -torch.norm(X - TN_G) ** 2, TN_G, tr_cores


def compute_linf_bound(config: Any, tensor: torch.Tensor) -> float:
    eps = config.epsilon if getattr(config, "epsilon", None) is not None else 0.0
    if eps <= 0:
        return 0.0
    return math.sqrt(eps) / (tensor.numel() ** 0.5)


def optimize_perturbation_linf(
    X: torch.Tensor,
    G_list_ori: List[torch.Tensor],
    config: Any,
    linf_bound: float,
    initial_E: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    attack_lr = getattr(config, "attack_lr", getattr(config, "lr", 0.01))
    if initial_E is not None:
        E = initial_E.clone().detach().to(X.device)
    else:
        E = torch.empty_like(X).uniform_(-linf_bound, linf_bound)
    E.requires_grad_(True)

    optimizer = torch.optim.Adam([E], lr=attack_lr)
    loss_history: List[float] = []

    for step in range(config.outer_num_steps_attack):
        optimizer.zero_grad()
        loss, TN_G, _ = objective_function_ATTR(X, G_list_ori, E, config.inner_num_steps, config.inner_tol)
        if torch.isnan(loss):
            raise ValueError("[Linf] Loss became NaN.")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            E.data.clamp_(-linf_bound, linf_bound)
        recon_error = torch.norm(X - TN_G).item()
        loss_history.append(recon_error)
        logging.info(
            f"[Linf-AdaTR] Step {step + 1}/{config.outer_num_steps_attack}, recon_err={recon_error:.6f}, "
            f"||E||_inf={torch.max(torch.abs(E)).item():.6f}"
        )
    with torch.no_grad():
        final_loss, final_recon, _ = objective_function_ATTR(X, G_list_ori, E, config.inner_num_steps, config.inner_tol)
        _ = final_loss
    return E.detach(), final_recon.detach(), loss_history


def optimize_perturbation(
    X: torch.Tensor,
    G_list_ori: List[torch.Tensor],
    config: Any,
    initial_E: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    loss_list: List[float] = []
    if initial_E is not None:
        E = initial_E.clone().detach().to(X.device)
        E.requires_grad_(True)
    else:
        E = torch.randn_like(X, requires_grad=True, device='cuda')

    scaling_factor = (config.epsilon ** 0.5) / torch.norm(E)
    if torch.norm(E) ** 2 > config.epsilon:
        E.data = scaling_factor * E

    loss_values: List[float] = []
    attack_lr = getattr(config, "attack_lr", getattr(config, "lr", 0.01))
    optimizer = torch.optim.Adam([E], lr=attack_lr)

    for step in range(config.outer_num_steps_attack):
        optimizer.zero_grad()
        E_old = E.clone().detach()
        X.requires_grad_(True)

        loss, TN_G, G_list = objective_function_ATTR(X, G_list_ori, E, config.inner_num_steps, config.inner_tol)

        if torch.isnan(loss):
            raise ValueError("Loss is NaN")

        if E.grad is not None:
            print("反向传播之前的梯度：", E.grad)
        loss.backward()

        print(f"Gradient max: {E.grad.max()}, min: {E.grad.min()}")
        if torch.isnan(E.grad).any():
            break

        optimizer.step()

        scaling_factor = (config.epsilon ** 0.5) / torch.norm(E)
        if torch.norm(E) ** 2 > config.epsilon:
            E.data = scaling_factor * E

        if torch.isnan(E).any():
            raise ValueError("E contains NaN after update")

        rse_E = torch.norm(E - E_old).item()
        recon_norm = torch.norm(X - TN_G).item()
        loss_list.append(recon_norm)

        norm_E = torch.norm(E).item()
        loss_value = loss.item()
        loss_sqrt = math.sqrt(abs(loss_value))
        loss_values.append(loss_sqrt)
        logging.info(
            f"[Ours] Step {step + 1}/{config.outer_num_steps_attack}, "
            f"loss={loss_value:.6f}, recon_norm={recon_norm:.6f}, ||E||={norm_E:.6f}, RSE_E={rse_E:.6f}"
        )
        print(
            f"[Ours] Step {step + 1}/{config.outer_num_steps_attack}: "
            f"loss={loss_value:.6f}, recon_norm={recon_norm:.6f}, ||E||={norm_E:.6f}, RSE_E={rse_E:.6f}"
        )

        prev_norm = torch.norm(E_old)
        change_norm = torch.norm(E.detach() - E_old)
        rel_change = change_norm / (prev_norm + 1e-12) if prev_norm > 0 else change_norm
        if rel_change < config.outer_tol:
            print(f"[Ours] Early stop at step {step + 1} due to relative E change {rel_change:.3e} < {config.outer_tol}")
            logging.info(f"[Ours] Early stop at step {step + 1} because relative E change={rel_change:.3e} < {config.outer_tol}")
            break

    return E.detach(), TN_G.detach(), loss_list


def optimize_perturbation_simplifed(
    X: torch.Tensor,
    G_list_ori: List[torch.Tensor],
    config: Any,
    initial_E: Optional[torch.Tensor] = None,
    linf_bound: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    rse_E_list: List[float] = []
    loss_list: List[float] = []
    attack_lr = getattr(config, "attack_lr", getattr(config, "lr", 0.01))

    if initial_E is not None:
        E = initial_E.clone().detach().to(X.device)
    else:
        E = torch.randn_like(X, device='cuda')
    if linf_bound is not None:
        E.data.clamp_(-linf_bound, linf_bound)

    total_size = sum(X.size())
    weights = [X.size(dim) / total_size for dim in range(X.dim())]

    scaling_factor = (config.epsilon ** 0.5) / torch.norm(E)
    if torch.norm(E) ** 2 > config.epsilon:
        E.data = (scaling_factor) * E

    for step in range(config.outer_num_steps_attack):
        G_list = tr.tr_decompose2(
            X + E,
            G_list_ori,
            max_iter=config.inner_num_steps,
            tol=config.inner_tol,
            dis_num=config.inner_dis_num,
            device="cuda",
        )
        TN_G = tl.tr_to_tensor(G_list)

        E_old = E.clone().detach()
        E_sum = torch.zeros_like(E)
        for n in range(X.dim()):
            X_dims = list(range(X.dim()))
            new_X_dims = [X_dims[n]] + X_dims[n + 1:] + X_dims[:n]
            X_unfold = tr.generalized_k_unfolding(X, new_X_dims, 0)

            X_hat = tr.contract_except_n(G_list, n)
            dims = list(range(X_hat.dim()))
            new_dims = [dims[-1]] + dims[:-1]
            Gr_unfold = tr.generalized_k_unfolding(X_hat, new_dims, 1)

            E_unfold = tr.generalized_k_unfolding(E, new_X_dims, 0)
            E_unfold_grad = (X_unfold - (X_unfold + E_unfold) @ torch.linalg.pinv(Gr_unfold) @ Gr_unfold) @ Gr_unfold.T @ torch.linalg.pinv(Gr_unfold).T
            E_unfold = E_unfold - attack_lr * E_unfold_grad

            E_folded = E_unfold.reshape(*[X.shape[dim] for dim in new_X_dims])
            E = E_folded.permute(*[new_X_dims.index(i) for i in range(len(new_X_dims))])
            E_sum += weights[n] * E
        E = E_sum
        if linf_bound is not None:
            E = torch.clamp(E, -linf_bound, linf_bound)

        scaling_factor = (config.epsilon ** 0.5) / torch.norm(E)
        if torch.norm(E) ** 2 > config.epsilon:
            E.data = (scaling_factor) * E
        if linf_bound is not None:
            E.data.clamp_(-linf_bound, linf_bound)

        rse_E = torch.norm(E - E_old).item()

        recon_norm = torch.norm(X - TN_G).item()
        noise_norm = torch.norm(E).item()
        loss_list.append(recon_norm)

        logging.info(
            f"[Ours_F] Step {step + 1}/{config.outer_num_steps_attack}, "
            f"loss≈{recon_norm:.6f}, recon_norm={recon_norm:.6f}, ||E||={noise_norm:.6f}, RSE_E={rse_E:.6f}"
        )
        print(
            f"[Ours_F] Step {step + 1}/{config.outer_num_steps_attack}: "
            f"loss≈{recon_norm:.6f}, recon_norm={recon_norm:.6f}, ||E||={noise_norm:.6f}, RSE_E={rse_E:.6f}"
        )

        rse_E_list.append(rse_E)

        prev_norm = torch.norm(E_old)
        rel_change = rse_E / (prev_norm + 1e-12) if prev_norm > 0 else rse_E
        if rel_change < config.outer_tol:
            print(f"[Ours_F] Early stop at step {step + 1} due to relative E change {rel_change:.3e} < {config.outer_tol}")
            logging.info(f"[Ours_F] Early stop at step {step + 1} because relative E change={rel_change:.3e} < {config.outer_tol}")
            break

        E = E.detach()
        TN_G = TN_G.detach()
        G_list = [G.detach() for G in G_list]

    return E, TN_G, loss_list


def optimize_perturbation_comparison(
    attacked_tensor: torch.Tensor,
    G_list: List[torch.Tensor],
    config: Any,
    attr_epsilon: Optional[float] = None,
    clean_tensor: Optional[torch.Tensor] = None,
    linf_bound: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """
    ATTR defence: starting from an attacked tensor, learn a bounded correction tensor E.
    """
    epsilon_sq = attr_epsilon if attr_epsilon is not None else config.attr_epsilon
    if epsilon_sq is None:
        epsilon_sq = config.epsilon

    attacked_tensor = attacked_tensor.detach()
    tr_cores = [core.clone().detach() for core in G_list]

    E = torch.zeros_like(attacked_tensor, device=attacked_tensor.device, requires_grad=True)
    defence_lr = getattr(config, "defence_lr", getattr(config, "lr", 0.01))
    optimizer = torch.optim.Adam([E], lr=defence_lr)

    loss_curve: List[float] = []

    prev_recon = None
    for step in range(config.outer_num_steps_defence):
        optimizer.zero_grad()

        E_prev = E.detach().clone()
        cores_prev = [core.clone().detach() for core in tr_cores]
        tr_cores = tr.tr_decompose2(
            attacked_tensor + E,
            tr_cores,
            max_iter=config.inner_num_steps,
            tol=config.inner_tol,
            dis_num=config.inner_dis_num,
            device=attacked_tensor.device,
        )
        tr_cores = [core.detach() for core in tr_cores]

        TN_G_attr = tl.tr_to_tensor(tr_cores)
        loss = torch.norm(attacked_tensor + E - TN_G_attr) ** 2
        loss = -loss
        loss.backward()

        optimizer.step()

        if epsilon_sq is not None:
            norm_E_sq = torch.norm(E.detach()) ** 2
            if norm_E_sq > epsilon_sq and norm_E_sq > 0:
                scale = (epsilon_sq / (norm_E_sq + 1e-12)) ** 0.5
                E.data.mul_(scale)
        current_norm_E = torch.norm(E.detach()).item()

        recon_err = torch.norm(attacked_tensor + E.detach() - TN_G_attr.detach()).item()
        loss_curve.append(recon_err)
        clean_diff = None
        if clean_tensor is not None:
            clean_diff = torch.norm(clean_tensor - TN_G_attr.detach()).item()
        log_msg = f"[ATTR] Step {step + 1}/{config.outer_num_steps_defence}: loss={loss.item():.6f}"
        if clean_diff is not None:
            log_msg += f", recon_norm_clean={clean_diff:.6f}"
        log_msg += f", recon_norm_attack={recon_err:.6f}, ||E||={current_norm_E:.6f}"
        print(log_msg)
        logging.info(log_msg)
        prev_norm_E = torch.norm(E_prev)
        change_norm_E = torch.norm(E.detach() - E_prev)
        rel_change_E = change_norm_E / (prev_norm_E + 1e-12) if prev_norm_E > 0 else change_norm_E
        rel_change_cores = 0.0
        for old_core, new_core in zip(cores_prev, tr_cores):
            delta = torch.norm(new_core - old_core)
            base = torch.norm(old_core)
            rel = delta / (base + 1e-12) if base > 0 else delta
            rel_change_cores = max(rel_change_cores, rel)
        rel_change = max(rel_change_E, rel_change_cores)
        if rel_change < config.outer_tol:
            print(f"[ATTR] Early stop at step {step + 1} due to relative change {rel_change:.3e} < {config.outer_tol}")
            logging.info(f"[ATTR] Early stop at step {step + 1} because relative change={rel_change:.3e} < {config.outer_tol}")
            break
        prev_recon = recon_err

    TN_G_attr = tl.tr_to_tensor(tr_cores).detach()
    return E.detach(), TN_G_attr, loss_curve
