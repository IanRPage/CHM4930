import torch
from torch import nn

from mufu.contract import (
    LOG_VAR_INIT,
    TASK_SPECS,
    TASKS,
    TaskLosses,
    _check_keys,
    _check_tensor,
)

LOG_VAR_MIN = -5.0
LOG_VAR_MAX = 5.0


class UncertaintyWeighting(nn.Module):
    """
    Learned homoscedastic uncertainty weighting across MuFu tasks (Kendall et al., 2018).

    Parameterizes each task k in TASKS with a learnable log-variance s_k = log(sigma_k^2):
        L_total = sum_{k present} [ c_k * exp(-s_k) * L_k + 0.5 * s_k ]
    where c_k = TASK_SPECS[k].loss_scale (0.5 for regression, 1.0 for classification),
    and s_k is clamped to [min_log_var, max_log_var] for numerical stability. Tasks whose
    batch loss is None are skipped entirely (0.5 * s_k term too).
    """

    def __init__(
        self,
        init_log_var: float = LOG_VAR_INIT,
        min_log_var: float = LOG_VAR_MIN,
        max_log_var: float = LOG_VAR_MAX,
    ) -> None:
        super().__init__()
        if min_log_var > max_log_var:
            raise ValueError(
                f"min_log_var ({min_log_var}) must be <= max_log_var ({max_log_var})"
            )
        self.min_log_var = float(min_log_var)
        self.max_log_var = float(max_log_var)
        self.log_vars = nn.ParameterDict(
            {task: nn.Parameter(torch.tensor(float(init_log_var))) for task in TASKS}
        )

    def forward(self, task_losses: TaskLosses) -> torch.Tensor:
        _check_keys("task_losses", set(task_losses), set(TASKS))

        terms: list[torch.Tensor] = []
        for task in TASKS:
            loss = task_losses[task]
            if loss is None:
                continue
            _check_tensor(task, loss, (), "float")
            if not torch.isfinite(loss):
                raise ValueError(f"{task}: loss is NaN or Inf")
            if loss < 0:
                raise ValueError(
                    f"{task}: loss must be non-negative, got {loss.item()}"
                )

            s_k = self.log_vars[task].clamp(self.min_log_var, self.max_log_var)
            c_k = TASK_SPECS[task].loss_scale
            terms.append(c_k * torch.exp(-s_k) * loss + 0.5 * s_k)

        if not terms:
            return torch.stack([p * 0.0 for p in self.log_vars.values()]).sum()

        return torch.stack(terms).sum()
