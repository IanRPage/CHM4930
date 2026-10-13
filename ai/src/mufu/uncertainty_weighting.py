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


class UncertaintyWeighting(nn.Module):
    """
    Learned homoscedastic uncertainty weighting across MuFu tasks (Kendall et al., 2018).

    Parameterizes each task k in TASKS with a learnable log-variance s_k = log(sigma_k^2):
        L_total = sum_{k present} [ c_k * exp(-s_k) * L_k + 0.5 * s_k ]
    where c_k = TASK_SPECS[k].loss_scale (0.5 for regression, 1.0 for classification).
    Tasks whose batch loss is None are skipped entirely (0.5 * s_k term too).
    """

    def __init__(self, init_log_var: float = LOG_VAR_INIT) -> None:
        super().__init__()
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
            if loss.isnan().any():
                raise ValueError(f"{task}: loss is NaN")

            s_k = self.log_vars[task]
            c_k = TASK_SPECS[task].loss_scale
            terms.append(c_k * torch.exp(-s_k) * loss + 0.5 * s_k)

        if not terms:
            return torch.stack([p * 0.0 for p in self.log_vars.values()]).sum()

        return torch.stack(terms).sum()
