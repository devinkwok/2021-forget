import math
import typing
import numpy as np


NO_CLASS_IDX = -1


def top_k(
        output_prob: np.ndarray, tgt_labels: np.ndarray, k=5
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Turns model outputs into a list of the top k scoring classes.

    Args:
        output_prob (np.ndarray): concatenated tensor of output probabilities (after Softmax),
            dimensions $(N \times C)$ where $N$ is number of examples, $C$ is number of classes.
        tgt_labels (np.ndarray): true classification labels given as
            integers between $0$ and $C - 1$ in a tensor of dimension $(N)$.
        k (int, optional): maximum number of classes to include. Defaults to 5.

    Returns:
        Tuple[np.ndarray, np.ndarray]: classification class indexes and probabilities
            for samples in output_prob, ordered by largest probability.
            Dimension of each array is $N \times k$.
            Probabilities are multiplied by -1 if class differs from tgt_labels.
    """
    n = output_prob.shape[0]
    top_k_classes = np.flip(np.argsort(output_prob, axis=1), axis=1)[:, :k]
    idx = np.arange(n).reshape(n, 1).repeat(k, axis=1)
    scores = output_prob[idx, top_k_classes]
    tgt_labels = tgt_labels.reshape(n, 1).repeat(k, axis=1)
    scores[top_k_classes != tgt_labels] *= -1
    return top_k_classes, scores


def n_cumulative_top_classes(top_classes_by_iter: typing.List[np.ndarray]) -> np.ndarray:
    """Counts the cumulative number of unique top classes for outputs of
    outputs_to_top_classes over multiple iterations.

    Args:
        top_classes (np.ndarray): list of outputs from outputs_to_top_classes
            over multiple iterations, where each iteration is
            called on same examples in same batch order.
            List should be ordered from earliest to latest iteration.

    Returns:
        np.ndarray: the number of unique classes assigned to each sample as iterations increase.
            Dimensions are $(N \times T)$ where $T$ is number of iterations.
    """
    n_cumulative = []
    cumulative = np.full_like(top_classes_by_iter[0], NO_CLASS_IDX)
    for top_classes in top_classes_by_iter:
        cumulative = np.concatenate([cumulative, top_classes], axis=1)
        n_unique = [len(np.unique(sample)) for sample in cumulative]
        n_cumulative.append(np.array(n_unique) - 1)  # don't count NO_CLASS_IDX
    return np.stack(n_cumulative, axis=1)


def top_1_auc(output_prob_by_iter: typing.List[np.ndarray], divide_by_iters=True) -> np.ndarray:
    """
    Args:
        output_prob_by_iter (typing.List[np.ndarray]): list of outputs from
            outputs_to_correctness over multiple iterations.
        divide_by_iters (bool, optional): If True, divide area under curve
            by number of iterations to normalize between 0 and 1. Defaults to True.

    Returns:
        np.ndarray: array of $N$ area under curve floats.
    """
    output_prob_by_iter = np.stack(output_prob_by_iter, axis=0)
    auc = np.sum(np.abs(output_prob_by_iter), axis=0)
    if divide_by_iters:
        auc = auc / output_prob_by_iter.shape[0]
    return auc


def diff_norm(output_prob_by_iter: typing.List[np.ndarray], norm_power=1) -> np.ndarray:
    """Takes difference of output probabilities between successive iterations,
    and calculates a norm on this differencing operation.

    Args:
        output_prob_by_iter (typing.List[np.ndarray]): list of outputs from
            outputs_to_correctness over multiple iterations,
            where each iteration is called on same examples in same batch order.
            List should be ordered from earliest to latest iteration.
        norm_power (int, optional): Type of norm (e.g. 1 is abs diff, 2 is mean squared difference).
            If less than 1, returns infinity norm (max difference). Defaults to 1.

    Returns:
        np.ndarray: array of $N$ mean differences for each sample over all iterations.
    """
    output_prob_by_iter = np.abs(np.stack(output_prob_by_iter, axis=0))
    diff = np.abs(output_prob_by_iter[:-1, :] - output_prob_by_iter[1:, :])
    if norm_power <= 0:
        return np.max(diff, axis=0)
    else:
        return (np.sum(diff**norm_power, axis=0) / diff.shape[0])**(1 / norm_power)
