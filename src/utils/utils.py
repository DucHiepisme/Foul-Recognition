import warnings
import torch
from omegaconf import DictConfig
import math

from importlib.util import find_spec
from torch._utils import _accumulate
import torch.utils
from torch import Generator, default_generator, randperm
from typing import (
    Any, 
    Callable, 
    Dict, 
    Optional, 
    Tuple,
    Sequence,
    List
)

import torch.utils.data

from src.utils import pylogger, rich_utils
from src.data.components.foul_dataset import Foul_Dataset
from src.utils.datasets import Subset

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value

def random_split(dataset: Foul_Dataset, lengths: Sequence[int],
                 generator: Optional[Generator] = default_generator,
                 split_by_label: Optional[bool] = True):

    if type(lengths[-1]).__name__ == "int" or not split_by_label:
        return torch.utils.data.random_split(
            dataset=dataset,
            lengths=lengths,
            generator=torch.Generator().manual_seed(42)
        )

    label_indexes = {}
    for i in range(len(dataset)):
        label = dataset.__getitem__(i)[-1].item()
        if label not in label_indexes.keys():
            label_indexes[label] = []
        
        label_indexes[label].append(i)
    
    subset_get_indexes: Dict[int, List[int]] = {}
    for label, indexes in label_indexes.items():
        subset_lengths = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(indexes) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)

        remainder = len(indexes) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        
        for i, length in enumerate(subset_lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")        

        random_indexes = randperm(sum(subset_lengths), generator=generator).tolist()  # type: ignore[call-overload]

        for i, (offset, length) in enumerate(list(zip(_accumulate(subset_lengths), subset_lengths))):
            if i not in subset_get_indexes.keys():
                subset_get_indexes[i] = []
            for index in random_indexes[offset - length: offset]:
                subset_get_indexes[i].append(indexes[index])

    return [Subset(dataset, indexes) for label, indexes in subset_get_indexes.items()]