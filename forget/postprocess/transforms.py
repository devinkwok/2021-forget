import typing
import torch
import numpy as np

NO_CLASS_IDX = -1


def identity(array):
    """For stacking using transform()
    """
    return array


def softmax(logits):
    with torch.no_grad():
        return torch.softmax(logits, dim=-1).detach().numpy()


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


def signed_prob(output_prob: np.ndarray, tgt_labels: np.ndarray) -> np.ndarray:
    """
    Args:
        output_prob (np.ndarray): output probabilities (after Softmax) with
            dimensions $(\dots, N \times C)$ where $N$ is number of examples,
            $C$ is number of classes.
        tgt_labels (np.ndarray): true classification labels given as
            integers between $0$ and $C - 1$ in a tensor of dimension $(N)$.

    Returns:
        (np.ndarray): probability of the correct class from `tgt_labels`,
            multiplied by -1 if the correct class was not the argmax,
            dimension $(\dots, N)$.
    """
    probabilities = output_prob[..., np.arange(len(tgt_labels)), tgt_labels]
    class_labels = np.argmax(output_prob, axis=-1)
    correct_mask = (class_labels == tgt_labels) * 2 - 1
    signed_prob = probabilities * correct_mask
    return signed_prob


def mean_prob(output_prob_by_iter: np.ndarray, divide_by_iters=True) -> np.ndarray:
    """
    Args:
        output_prob_by_iter (np.ndarray): signed_prob scores
            with dimensions $(\dots, I \times N)$.
            where each iteration $I$ is over same examples in same batch order.
        divide_by_iters (bool, optional): If True, divide area under curve
            by number of iterations to normalize between 0 and 1. Defaults to True.

    Returns:
        np.ndarray: mean of $N$ over $I$, dimensions $(\dots, N)$.
    """
    auc = np.sum(np.abs(output_prob_by_iter), axis=-2)
    if divide_by_iters:
        auc = auc / output_prob_by_iter.shape[-2]
    return auc


def diff_norm(output_prob_by_iter: np.ndarray,
            norm_power=1, divide_by_iters=True) -> np.ndarray:
    """Takes difference of output probabilities between successive iterations,
    and calculates a norm on this differencing operation.

    Args:
        output_prob_by_iter (np.ndarray): signed_prob scores
            with dimensions $(\dots, I \times N)$,
            where each iteration $I$ is over same examples in same batch order.
        norm_power (int, optional): Type of norm (e.g. 1 is abs diff, 2 is mean squared difference).
            If less than 1, returns infinity norm (max difference). Defaults to 1.
        divide_by_iters (bool, optional): If True, divide result by $N$.

    Returns:
        np.ndarray: array of $(\dots, N)$ mean differences.
    """
    output_prob_by_iter = np.abs(output_prob_by_iter)
    diff = np.abs(output_prob_by_iter[..., :-1, :] - output_prob_by_iter[..., 1:, :])
    if norm_power <= 0:
        norm = np.max(diff, axis=-2)
    else:
        norm = (np.sum(diff**norm_power, axis=-2) / diff.shape[-2])**(1 / norm_power)
    if divide_by_iters:
        norm = norm / output_prob_by_iter.shape[-2]
    return norm


def forgetting_events(output_prob_by_iter: np.ndarray,
            iter_mask: np.ndarray=None) -> np.ndarray:
    """Counts number of forgetting events, as defined by Toneva et al., 2018.
    Specifically, an example is forgotten if it is incorrect at time t+1 and
    correct at time t.

    Args:
        output_prob_by_iter (np.ndarray): signed_prob scores
            with dimensions $(\dots, I \times N)$,
        iter_mask (np.ndarray, optional): Boolean mask of $(I \times N)$.
            For each example, forgetting events will only be detected
            between successive time steps which are True in the mask.
            Each example must have the same number of True values over I.
            Defaults to None.

    Returns:
        np.ndarray: array of $(\dots, N)$ forgetting counts.
    """
    if iter_mask is not None:
        assert output_prob_by_iter.shape[-1] == iter_mask.shape[-1]
        # mask must have same number of True sampling points for every example
        total = np.sum(iter_mask, axis=-2)
        n_unmasked = total.flatten()[0]
        assert np.all(total == n_unmasked)
        # repeat mask in I dim until it fills output_prob_by_iter
        assert output_prob_by_iter.shape[-2] % iter_mask.shape[-2] == 0
        n_repeats = output_prob_by_iter.shape[-2] // iter_mask.shape[-2]
        iter_mask = iter_mask.repeat(n_repeats, axis=-2)
        new_shape = list(output_prob_by_iter.shape)
        new_shape[-2] = n_unmasked * n_repeats
        output_prob_by_iter = output_prob_by_iter[..., iter_mask].reshape(new_shape)
    is_correct = (output_prob_by_iter > 0)  # use sign to indicate whether correct/incorrect
    diff = np.logical_and(is_correct[..., :-1, :], np.logical_not(is_correct[..., 1:, :]))
    n_forget = np.sum((diff == 1), axis=-2) # correct - incorrect is 1 - 0
    return n_forget


def mask_iter_by_batch(train_batch_size, n_train_examples,
            example_start_idx, example_end_idx) -> np.ndarray:
    """Generates iteration mask for when examples occur in the training batch.
    This mask allows forgetting_events to compute the same forgetting counts
    as calculated by Toneva et al., 2018.
    E.g. for a batch size of 128, the first 128 examples are unmasked at time t,
    then the next 128 at time t+1, etc.

    Args:
        train_batch_size (int): batch size in training
        n_train_examples (int): number of examples in train set (N)
        example_start_idx (int): index of first example to include
        example_end_idx (int): index of last example to include

    Returns:
        np.ndarray: array of $(I, N)$ True/False mask values
    """
    n_batch = int(np.ceil(n_train_examples / train_batch_size))
    iter_mask = np.zeros((n_batch, example_end_idx - example_start_idx), dtype=bool)
    iter_idx = np.arange(n_batch).repeat(train_batch_size)[example_start_idx:example_end_idx]
    example_idx = np.arange(example_end_idx - example_start_idx)
    iter_mask[iter_idx, example_idx] = True
    return iter_mask


def stats_str(array):
    """Helper for pretty printing
    """
    return '<{:0.4f}|{:0.4f}|{:0.4f}> {}'.format(
        np.min(array), np.mean(array), np.max(array), array.shape)
