from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from .combine_scheduler import CombineLRSchedulers


class WarmUpScheduler(CombineLRSchedulers):
    """
    Gradual WarmUp Learning Rate Scheduler proposed in
    `Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour` (https://arxiv.org/abs/1706.02677).

    Parameters
    ----------
    optimizer : `Optimizer`, required
        Wrapped optimizer.
    warmup_steps : `int`, required
        Number of steps for Gradual WarmUp stage.
    after_warmup_scheduler : `_LRScheduler`, required
        Scheduler after Gradual WarmUp.
    add_constant_steps : `int`, optional (default = `None`)
        Number of steps for stage with constant learning rate.
        If None constant stage is not considered.
    warmup_denominator : `int`, optional (default = `None`)
        Denominator for warmup like in `Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour`
        where learning rate was 0.1 * kn / 256. 256 is a denominator.
        With this for Gradual WarmUp stage at the end we could achieve
        learning rate higher than it was in `optimizer`.
        If None use `warmup_steps` as a denominator.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        after_warmup_scheduler: _LRScheduler,
        add_constant_steps: int = None,
        warmup_denominator: int = None,
    ) -> None:
        if add_constant_steps is not None and add_constant_steps <= 0:
            raise ValueError('add_constant_steps should be greater than 1.')
        if add_constant_steps:
            lr_schedulers = [
                LambdaLR(
                    optimizer,
                    lr_lambda=lambda step: step / max(1.0, warmup_denominator or warmup_steps),
                ),
                LambdaLR(optimizer, lr_lambda=lambda step: 1.0),
                after_warmup_scheduler
            ]
            lr_schedulers_steps = [warmup_steps, add_constant_steps]
        else:
            lr_schedulers = [
                LambdaLR(
                    optimizer,
                    lr_lambda=lambda step: step / max(1.0, warmup_denominator or warmup_steps),
                ),
                after_warmup_scheduler
            ]
            lr_schedulers_steps = [warmup_steps]
        super().__init__(
            lr_schedulers=lr_schedulers,
            lr_schedulers_steps=lr_schedulers_steps
        )
