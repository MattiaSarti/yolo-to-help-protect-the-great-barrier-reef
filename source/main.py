"""
Execution of the proposed competition solution.
"""


from random import seed as random_seed

from numpy.random import seed as numpy_seed
# pylint: disable=import-error
from tensorflow.random import set_seed
# pylint: enable=import-error


def fix_seeds_for_reproducible_results() -> None:
    """
    Make the subsequent instructions produce purely deterministic outputs by
    fixing all the relevant seeds.
    """
    random_seed(a=0)
    _ = numpy_seed(seed=0)
    set_seed(seed=0)


def main() -> None:
    """
    Execute the proposed competition solution.
    """
    fix_seeds_for_reproducible_results()

    raise NotImplementedError
