from typing import Type
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    _LRScheduler, LambdaLR,
    CosineAnnealingLR
)
from .combine_scheduler import CombineLRSchedulers


class WarmUpScheduler(CombineLRSchedulers):
    """
    Gradual WarmUp Learning Rate Scheduler proposed in
    `Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour`
    (https://arxiv.org/abs/1706.02677).

    Parameters
    ----------
    optimizer : `Optimizer`, required
        Wrapped optimizer.
    warmup_steps : `int`, required
        Number of steps for Gradual WarmUp stage.
    after_warmup_scheduler : `_LRScheduler`, required
        Scheduler after Gradual WarmUp.
    starts_with : `int`, optional (default = `None`)
        Initial learning rate to start WarmUp Stage.
        If None then starting point is considered to be
        `optimizer.lr / warmup_steps`.
    add_constant_steps : `int`, optional (default = `None`)
        Number of steps for stage with constant learning rate.
        If None constant stage is not considered.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        after_warmup_scheduler: _LRScheduler,
        starts_with: int = None,
        add_constant_steps: int = None
    ) -> None:
        # 0: Validate passed optional values
        for val in [starts_with, add_constant_steps]:
            self._validate(val)
        # 1: Configure WarmUp Stage
        warmup_steps = max(1.0, warmup_steps)
        if starts_with:
            # Use initial_lr as each instance of _LRScheduler makes its first (0 step) during init
            # and as we expect after_warmup_scheduler to be an instance of _LRScheduler then
            # we will definitely have initial_lr for optimizer.param_groups
            lr_start = starts_with / optimizer.param_groups[0].get('initial_lr', 1)
            num_steps = lr_start * warmup_steps + warmup_steps
        else:
            lr_start = 0
            num_steps = warmup_steps
        warmup = LambdaLR(
            optimizer,
            # Substract 1 / num_steps as we start from step = 1
            lr_lambda=lambda step: min(1.0, (step / num_steps) + max(0, lr_start - (1 / num_steps)))
        )
        # 2: Set Constant LR Stage after WarmUp if needed
        if add_constant_steps:
            const_lr = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
            lr_schedulers = [warmup, const_lr, after_warmup_scheduler]
            lr_schedulers_steps = [warmup_steps, add_constant_steps]
        else:
            lr_schedulers = [warmup, after_warmup_scheduler]
            lr_schedulers_steps = [warmup_steps]
        super().__init__(
            lr_schedulers=lr_schedulers,
            lr_schedulers_steps=lr_schedulers_steps
        )

    @classmethod
    def with_linear_stage(
        cls: Type['WarmUpScheduler'],
        optimizer: Optimizer,
        warmup_steps: int,
        num_training_steps: int,
        starts_with: int = None
    ) -> Type['WarmUpScheduler']:
        """
        Instantiate WarmUpScheduler with linear decreasing learning rate after it.

        Parameters
        ----------
        optimizer : `Optimizer`, required
            Wrapped optimizer.
        warmup_steps : `int`, required
            Number of steps for Gradual WarmUp stage.
        num_training_steps : `int`, required
            Number of steps in training phase.
        starts_with : `int`, optional (default = `None`)
            Initial learning rate to start WarmUp Stage.
            If None then starting point is considered to be
            `optimizer.lr / warmup_steps`.
        """
        return cls(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            after_warmup_scheduler=LambdaLR(
                optimizer,
                lr_lambda=lambda step: max(
                    0.0,
                    (num_training_steps - step) / max(1.0, num_training_steps - warmup_steps)
                )
            ),
            starts_with=starts_with
        )

    @classmethod
    def with_cosine_stage(
        cls: Type['WarmUpScheduler'],
        optimizer: Optimizer,
        warmup_steps: int,
        num_training_steps: int,
        starts_with: int = None
    ) -> Type['WarmUpScheduler']:
        """
        Instantiate WarmUpScheduler with cosine annealing learning rate after it.

        Parameters
        ----------
        optimizer : `Optimizer`, required
            Wrapped optimizer.
        warmup_steps : `int`, required
            Number of steps for Gradual WarmUp stage.
        num_training_steps : `int`, required
            Number of steps in training phase.
        starts_with : `int`, optional (default = `None`)
            Initial learning rate to start WarmUp Stage.
            If None then starting point is considered to be
            `optimizer.lr / warmup_steps`.
        """
        return cls(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            after_warmup_scheduler=CosineAnnealingLR(
                optimizer,
                T_max=num_training_steps
            ),
            starts_with=starts_with
        )

    @classmethod
    def with_constant_stage(
        cls: Type['WarmUpScheduler'],
        optimizer: Optimizer,
        warmup_steps: int,
        starts_with: int = None
    ) -> Type['WarmUpScheduler']:
        """
        Instantiate WarmUpScheduler with constant learning rate after it.

        Parameters
        ----------
        optimizer : `Optimizer`, required
            Wrapped optimizer.
        warmup_steps : `int`, required
            Number of steps for Gradual WarmUp stage.
        starts_with : `int`, optional (default = `None`)
            Initial learning rate to start WarmUp Stage.
            If None then starting point is considered to be
            `optimizer.lr / warmup_steps`.
        """
        return cls(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            after_warmup_scheduler=LambdaLR(optimizer, lr_lambda=lambda step: 1.0),
            starts_with=starts_with
        )

    @staticmethod
    def _validate(x: int) -> None:
        """Static function to validate `int` values passed to `init`."""
        if x is not None and x <= 0:
            raise ValueError(
                'starts_with or add_constant_steps'
                'should be greater than 0.'
            )
