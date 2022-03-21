import typing
import torch
import torch.nn.functional as F
import numpy as np

NO_CLASS_IDX = -1


def identity(array):
    """For stacking using transform()"""
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


def n_cumulative_top_classes(
    top_classes_by_iter: typing.List[np.ndarray],
) -> np.ndarray:
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
    sgn_prob = probabilities * correct_mask
    return sgn_prob


def margin(
    output_prob: np.ndarray,
    one_hot_labels: np.ndarray,
    ord=None,
    divide_by_classes=True,
) -> np.ndarray:
    """
    Args:
        output_prob (np.ndarray): output probabilities (after Softmax) with
            dimensions $(\dots, N \times C)$ where $N$ is number of examples,
            $C$ is number of classes.
        one_hot_labels (np.ndarray): true classification labels given as one hot
            vectors with components in $\{0, 1\}$ and dimension $(N, C)$.
        ord (int): order for np.linalg.norm. Default is None (L2 norm).
        divide_by_classes (bool): if True, normalize norm by dividing by $C$.
            Default is True.

    Returns:
        (np.ndarray): norm of difference between one_hot_labels and output_prob.
            Dimensions are $(\dots, N)$.
    """
    assert output_prob.shape[-2:] == one_hot_labels.shape
    margin = np.linalg.norm(output_prob - one_hot_labels, ord=ord, axis=-1)
    if divide_by_classes:
        return margin / output_prob.shape[-1]
    return margin


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


def diff_norm(
    output_prob_by_iter: np.ndarray, norm_power=1, divide_by_iters=True
) -> np.ndarray:
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
        norm = (np.sum(diff**norm_power, axis=-2) / diff.shape[-2]) ** (
            1 / norm_power
        )
    if divide_by_iters:
        norm = norm / output_prob_by_iter.shape[-2]
    return norm


def forgetting_events(
    output_prob_by_iter: np.ndarray,
    iter_mask: np.ndarray = None,
    never_learned_value: int = None,
    return_count=True,
) -> np.ndarray:
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
        never_learned_value (int, optional): If set, any example in dimension N
            which never has a learning event is set to this count.
            Alternatively, if return_count is False, create a forgetting event at this
            iteration in I for examples which are never learned, otherwise,
            create a forgetting event at the last iteration for examples
            which are never learned. Defaults to None.
        return_count (bool): If True, return count over I, otherwise return an array
            with 0, 1 over dimension I indicating if forgetting occurred. Defaults to True.

    Returns:
        np.ndarray: array of $(\dots, N)$ forgetting counts.
    """
    if iter_mask is not None:
        assert len(iter_mask.shape) == 2
        assert output_prob_by_iter.shape[-1] == iter_mask.shape[-1]
        # mask must have same number of True sampling points for every example
        total = np.sum(iter_mask, axis=-2)
        n_unmasked = total.flatten()[0]
        assert np.all(total == n_unmasked)
        # repeat mask in I dim until it fills output_prob_by_iter
        assert output_prob_by_iter.shape[-2] % iter_mask.shape[-2] == 0
        n_repeats = output_prob_by_iter.shape[-2] // iter_mask.shape[-2]
        iter_mask = (
            np.expand_dims(iter_mask, axis=0)
            .repeat(n_repeats, axis=0)
            .reshape(-1, iter_mask.shape[-1])
        )
        new_shape = list(output_prob_by_iter.shape)
        new_shape[-2] = n_unmasked * n_repeats
        output_prob_by_iter = output_prob_by_iter[..., iter_mask].reshape(new_shape)
    is_correct = (
        output_prob_by_iter > 0
    )  # use sign to indicate whether correct/incorrect
    diff = np.logical_and(
        is_correct[..., :-1, :], np.logical_not(is_correct[..., 1:, :])
    )
    forget_events = diff == 1
    never_learned = np.logical_not(np.any(is_correct, axis=-2))
    if not return_count:
        # add one forget_event to last iter of never learned examples
        if never_learned_value is None:
            never_learned_value = -1
        forget_events[..., never_learned_value, :][never_learned] = True
        return forget_events
    n_forget = np.sum(forget_events, axis=-2)  # correct - incorrect is 1 - 0
    # set examples which were never learned to some value
    if never_learned_value is None:
        never_learned_value = np.max(n_forget) + 1
    # never_learned is 0 forgetting AND incorrect at last iter
    n_forget[never_learned] = never_learned_value
    return n_forget


def mask_iter_by_batch(
    train_batch_size, n_train_examples, example_start_idx, example_end_idx
) -> np.ndarray:
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
    iter_idx = np.arange(n_batch).repeat(train_batch_size)[
        example_start_idx:example_end_idx
    ]
    example_idx = np.arange(example_end_idx - example_start_idx)
    iter_mask[iter_idx, example_idx] = True
    return iter_mask


def last_always_true_index(boolean_mask: np.ndarray, scale=None) -> np.ndarray:
    """Finds first index after which boolean_mask is always True

    Args:
        boolean_mask (np.ndarray): mask of boolean criterion for each index
            with dimensions $(\dots, I \times N)$.
        scale (np.ndarray, optional): If set, map indexes to
            these values. Has dimension (I+1). Defaults to None.

    Returns:
        np.ndarray: array of $(\dots, N)$ index or
            scale value after which boolean_mask is always True.
    """
    # cumulative product is 1 until False, then 0
    # add up all 1's to get iter of first True value
    # 0 if all incorrect, shape[-2] if all correct
    first = np.sum(np.cumprod(boolean_mask, axis=-2), axis=-2)
    if scale is not None:
        assert scale.shape == (boolean_mask.shape[-2] + 1,)
        first = np.take(scale, first)
    return first


def first_always_true_index(boolean_mask: np.ndarray, scale=None) -> np.ndarray:
    """Finds last index before which boolean_mask is always True

    Args:
        boolean_mask (np.ndarray): mask of boolean criterion for each index
            with dimensions $(\dots, I \times N)$.
        scale (np.ndarray, optional): If set, map indexes to
            these values. Has dimension (I+1). Defaults to None.

    Returns:
        np.ndarray: array of $(\dots, N)$ index or
            scale value after which boolean_mask is always True.
    """
    if scale is None:
        scale = np.arange(boolean_mask.shape[-2] + 1)
    return last_always_true_index(np.flip(boolean_mask, axis=-2), scale=np.flip(scale))


def stats_str(array):
    """Helper for pretty printing"""
    return "<{:0.4f}|{:0.4f}/{:0.4f}|{:0.4f}> {}".format(
        np.min(array), np.mean(array), np.std(array), np.max(array), array.shape
    )


def intersection_over_union(set_1: np.ndarray, set_2: np.ndarray) -> float:
    """Computes jaccard index (intersection over union) of two bool masks of broadcastable dimensions.

    Args:
        set_1 (np.ndarray): bool mask
        set_2 (np.ndarray): bool mask

    Returns:
        float: sum(set_1 AND set_2) / sum(set_1 OR set_2)
    """
    return np.sum(np.logical_and(set_1, set_2)) / np.sum(np.logical_or(set_1, set_2))


def jaccard_similarity(
    set_1: np.ndarray, set_2: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes jaccard index while varying threshold of inclusion from 0 to 100%.

    Args:
        set_1 ([np.ndarray]): set of values to threshold, dimension (N)
        set_2 ([np.ndarray]): another set of values to threshold, dimension (N)

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: jaccard indexes, thresholds for set_1,
            and thresholds for set_2, all dimension N
    """
    assert (
        len(set_1.shape) == 1 and len(set_2.shape) == 1 and set_1.shape == set_2.shape
    )
    idx_1 = np.argsort(set_1)
    idx_2 = np.argsort(set_2)
    mask_1 = np.zeros_like(set_1, dtype=bool)
    mask_2 = np.zeros_like(set_2, dtype=bool)
    jaccard = []
    for i in range(len(set_1)):
        mask_1[idx_1[i]] = True
        mask_2[idx_2[i]] = True
        jaccard += [intersection_over_union(mask_1, mask_2)]
    return np.array(jaccard), set_1[idx_1], set_2[idx_2]


def center_of_mass(weight: np.ndarray, index: np.ndarray = None, normalize=True):
    """Computes center of mass of index in array weighted by array values.

    Args:
        weight (np.ndarray): weight of each index, dimension (..., I \times N)
        index (np.ndarray, optional): If None, weights are multiplied by the index (1, 2, \dots I).
            Otherwise weights are multiplied by this array of dimension I. Defaults to None.
        normalize (bool, optional): If True, divide result by sum of weights. Defaults to True.

    Returns:
        np.ndarray: dot product of weight and index of dimension (..., N)
    """
    if index is None:
        index = np.arange(weight.shape[-2])
    assert index.shape == (weight.shape[-2],), index.shape
    center_of_mass = np.tensordot(weight, index, axes=([-2], [0]))
    if normalize:
        center_of_mass = center_of_mass / np.sum(weight, axis=-2)
    return center_of_mass


def moving_average(input: np.ndarray, window_len: int, pad_ends=False):
    """Computes moving average over 2nd last dimension (time or iterations) of input.

    Args:
        input (np.ndarray): array of dimension (..., I \times N).
        window_len (int): number of iterations/timesteps to compute moving average over.

    Returns:
        np.ndarray: moving average of dimension (..., I - window_len + 1, N)
    """
    running_total = np.cumsum(input, axis=-2)
    average = running_total[..., window_len:, :] - running_total[..., :-window_len, :]
    # running total does not start from 0, include point at window_len - 1
    average = np.concatenate(
        [running_total[..., window_len - 1 : window_len, :], average], axis=-2
    )
    if pad_ends:
        n_pad_start = (input.shape[-2] - average.shape[-2]) // 2
        n_pad_end = input.shape[-2] - average.shape[-2] - n_pad_start
        start_pad = np.repeat(average[..., :1, :], n_pad_start, axis=-2)
        end_pad = np.repeat(average[..., -1:, :], n_pad_end, axis=-2)
        average = np.concatenate([start_pad, average, end_pad], axis=-2)
    return average / window_len
