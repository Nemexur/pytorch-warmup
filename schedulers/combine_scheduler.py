from typing import (
    List, Dict, Any
)
from torch.optim.lr_scheduler import _LRScheduler


class CombineLRSchedulers:
    """
    Combine several Learning Rate Schedulers with certain
    number of steps for each pair of `lr_schedulers`.

    Parameters
    ----------
    lr_schedulers : `List[_LRScheduler]`, required
        List of PyTorch._LRScheduler for combining.
    lr_schedulers_steps : `List[int]`, required
        Steps for each pair of _LRScheduler.
        It should be one less than len(lr_schedulers).
    """
    def __init__(
        self,
        lr_schedulers: List[_LRScheduler],
        lr_schedulers_steps: List[int]
    ) -> None:
        if not all(isinstance(x, _LRScheduler) for x in lr_schedulers):
            raise TypeError(
                'All schedulers in lr_schedulers should be subclasses of _LRScheduler.'
            )
        if len(lr_schedulers) - len(lr_schedulers_steps) != 1:
            raise ValueError(
                'Number of lr_schedulers_steps should be one less '
                'than number of lr_schedulers as we are working with pairs.'
            )
        self.lr_schedulers = lr_schedulers
        self._lr_schedulers_steps = lr_schedulers_steps
        self._last_step = 0
        self._reached_end = False
        self._lr_scheduler_idx = 0
        self._current_num_steps = 0

    @property
    def current_lr_scheduler(self):
        return self.lr_schedulers[self._lr_scheduler_idx]

    def state_dict(self) -> Dict[str, Any]:
        """
        Returns the state of the scheduler as a `dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the lr_schedulers.
        """
        return {
            key: value for key, value in self.__dict__.items()
            if key != 'lr_schedulers'
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Loads the schedulers state.

        Parameters
        ----------
        state_dict : `Dict[str, Any]`, required
            Scheduler state. Should be an object returned
            from a call to `state_dict`.
        """
        self.__dict__.update(state_dict)

    def step(self, step: int = None, **kwargs) -> None:
        self._last_step += 1
        if not self._reached_end:
            self._update_if_needed()
        self.lr_schedulers[self._lr_scheduler_idx].step(step, **kwargs)

    def _update_if_needed(self) -> None:
        # As python slicing does not include value at index self._current_num_steps
        if self._last_step > sum(self._lr_schedulers_steps[:self._current_num_steps + 1]):
            # Sum here is pretty fast because usually you
            # do not combine more than 3 different LRSchedulers.
            previous_base_lrs = self.lr_schedulers[self._lr_scheduler_idx].base_lrs
            self._lr_scheduler_idx += 1
            self._current_num_steps += 1
            self.lr_schedulers[self._lr_scheduler_idx].base_lrs = previous_base_lrs
            # As we start from zero we should substract one
            self.lr_schedulers[self._lr_scheduler_idx].last_epoch = self._last_step - 1
        if self._lr_scheduler_idx == len(self.lr_schedulers) - 1:
            self._reached_end = True
