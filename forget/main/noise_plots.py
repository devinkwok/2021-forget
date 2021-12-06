import numpy as np
from forget.postprocess.transforms import stats_str


def rank(metric):
    """Turns metrics into ranks.

    Args:
        metric (np.ndarray): array of metrics with dimensions $(\dots, N)$

    Returns:
        np.ndarray: array with same dimensions as metric,
            with values replaced by rank over last dimension $N$.
    """
    sorted_idx = np.argsort(metric, axis=-1)
    rank_idx = np.arange(metric.shape[-1])
    ranks = np.empty_like(sorted_idx)
    np.put_along_axis(ranks, sorted_idx, rank_idx, axis=-1)
    return ranks


def apply(dict_metrics, transform_fn, suffix=""):
    if suffix != "":
        suffix = "-" + suffix
    output_metrics = {}
    for name, metric in dict_metrics.items():
        output_metrics[f"{name}{suffix}"] = transform_fn(metric)
    return output_metrics


def plot_noise_metrics_by_sample(plotter, noise_metrics):
    # noise_metrics have dims R x S x N
    def squeeze(metric):
        return metric.squeeze()

    """ Q: is it reasonable to rank examples by mean/median noise needed to perturb?
        for metric noise_first_forget
        A) scatter meanS/medianS to id
            look for close to diagonal and not too spread out
        B) scatter meanS to stdS
            if homogenous over means, low variance as a % of noise scale, then mean is representative
            this allows averaging out S (sample dimension)
    """
    meanS_noise = apply(
        noise_metrics,
        lambda metric: np.mean(metric, axis=-2, keepdims=True),
        suffix="mu",
    )
    medianS_noise = apply(
        noise_metrics,
        lambda metric: np.median(metric, axis=-2, keepdims=True),
        suffix="med",
    )
    stdS_noise = apply(
        noise_metrics,
        lambda metric: np.std(metric, axis=-2, keepdims=True),
        suffix="sd",
    )

    plotter.plot_array(
        plotter.plt_scatter, "noise_meanS_stdS", noise_metrics, meanS_noise
    )
    plotter.plot_array(
        plotter.plt_scatter, "noise_meanS_stdS", noise_metrics, medianS_noise
    )
    plotter.plot_array(plotter.plt_scatter, "noise_meanS_stdS", stdS_noise, meanS_noise)
    """ C) scatter meanS to medianS
            if these look diagonal, then median is equivalent to mean
    """
    plotter.plot_array(
        plotter.plt_scatter, "noise_meanS_medianS", medianS_noise, meanS_noise
    )
    plotter.plot_array(
        plotter.plt_pair_corr,
        "noise_meanS_medianS",
        apply(medianS_noise, squeeze),
        apply(meanS_noise, squeeze),
    )
    """ D) scatter rankNofmeanS/medianS to meanS/medianSofrankN
            if these look the same, then order of applying rank/mean doesn't matter
        A, B, C, D confirm use of median over S for rank corr with other metrics
    """
    rankofmedians = apply(
        noise_metrics,
        lambda metric: rank(np.median(metric, axis=-2, keepdims=True)),
        suffix="rkmed",
    )
    medianofranks = apply(
        noise_metrics,
        lambda metric: np.median(rank(metric), axis=-2, keepdims=True),
        suffix="medrk",
    )
    plotter.plot_array(
        plotter.plt_scatter, "noise_rank_median_order", medianofranks, rankofmedians
    )
    plotter.plot_array(
        plotter.plt_pair_corr,
        "noise_rank_median_order",
        apply(medianofranks, squeeze),
        apply(rankofmedians, squeeze),
    )
    # return R x N metrics summarized over S
    return apply(noise_metrics, lambda metric: np.median(metric, axis=-2))


def plot_comparisons(plotter, metrics):
    # choose 1 of mean or median
    meanR = apply(
        metrics, lambda metric: np.median(metric, axis=-2, keepdims=True), suffix="med"
    )
    """
    Q: are any metrics related?
        A) scatter identity to identity all metrics
            if any show strong trend, they are similar
    Q: is the variation in each metric due to inits or the examples themselves?
        A) corr identity to identity
        B) scatter/corr meanR to identity/rankN
            taking mean improves corr if it is due to examples themselves, reduces corr if it is due to inits
            compare A mean B identity and B mean A identity to tell whether metric A or B depends more on inits
        C) scatter/corr meanR to meanR all metrics
            this tells whether variance in metrics is due to inits (higher corr here, due to averaging out variance)
            or due to examples (reduced corr here, due to correlated variance which cannot be averaged out)
    """
    plotter.plot_array(plotter.plt_scatter, "noise_id_to_id", metrics, metrics)
    plotter.plot_array(plotter.plt_self_corr, "noise_id_to_id", metrics, metrics)
    plotter.plot_array(plotter.plt_pair_corr, "noise_id_to_id", metrics, metrics)

    plotter.plot_array(plotter.plt_scatter, "noise_mean_to_id", metrics, meanR)
    plotter.plot_array(plotter.plt_pair_corr, "noise_mean_to_id", metrics, meanR)

    plotter.plot_array(plotter.plt_scatter, "noise_mean_to_mean", meanR, meanR)
    plotter.plot_array(plotter.plt_pair_corr, "noise_mean_to_mean", meanR, meanR)
    """
    Q: is variance over inits correlated to any metrics?
        A) scatter/corr stdR to identity/rankN all metrics
        B) scatter/corr stdR to meanR all metrics
    """
    stdR = apply(
        metrics, lambda metric: np.std(metric, axis=-2, keepdims=True), suffix="std"
    )
    plotter.plot_array(plotter.plt_scatter, "noise_std_to_id", stdR, metrics)
    plotter.plot_array(plotter.plt_pair_corr, "noise_std_to_id", stdR, metrics)

    plotter.plot_array(plotter.plt_scatter, "noise_std_to_mean", stdR, meanR)
    plotter.plot_array(plotter.plt_pair_corr, "noise_std_to_mean", stdR, meanR)

    plotter.plot_array(plotter.plt_scatter, "noise_std_to_std", stdR, stdR)
    plotter.plot_array(plotter.plt_pair_corr, "noise_std_to_std", stdR, stdR)

def plots_2021_11_24(plotter, all_metrics, learned_before_iter_inclusive, always_learned):
    metrics = {
        'first_learn': all_metrics['first_learn'],
        'noise_first_forget': all_metrics['noise_first_forget'],
        'prune_first_forget': all_metrics['prune_first_forget'],
    }
    medianR = apply(
        metrics, lambda metric: np.median(metric, axis=-2, keepdims=True), suffix="med"
    )
    metrics = mask_learned(metrics, learned_before_iter_inclusive, always_learned)
    if always_learned:
        plot_prefix = 'allmaskto' + str(learned_before_iter_inclusive) + '_'
    else:
        plot_prefix = 'anymaskto' + str(learned_before_iter_inclusive) + '_'
    plotter.plot_array(plotter.plt_scatter, plot_prefix + 'id_to_id', metrics, metrics)
    #FIXME not broadcasting
    # plotter.plot_array(plotter.plt_scatter, 'id_to_med', metrics, medianR)
    # plotter.plot_array(plotter.plt_scatter, 'med_to_med', medianR, medianR)
    plotter.plot_array(plotter.plt_self_corr, plot_prefix + "id_to_id", metrics, metrics)
    plotter.plot_array(plotter.plt_pair_corr, plot_prefix + "id_to_id", metrics, metrics)

def mask_learned(metrics, learned_before_iter_inclusive=-1, always_learned=False):
    if learned_before_iter_inclusive < 0:
        return metrics
    mask = metrics['first_learn'] <= learned_before_iter_inclusive
    if always_learned:
        mask = np.all(mask, axis=-2)
    else:
        mask = np.any(mask, axis=-2)
    print('Include examples learned before iter',
        learned_before_iter_inclusive, stats_str(mask),
        'out of', stats_str(metrics['first_learn']))
    return apply(metrics, lambda metric: metric[..., mask],
                suffix=f'lrn={learned_before_iter_inclusive}')
