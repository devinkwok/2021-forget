import numpy as np
from forget.job import Job
from forget.postprocess.metrics import Metrics
from forget.postprocess.plot_metrics import PlotMetrics
from forget.postprocess.transforms import stats_str


class ExperimentPlots:
    def __init__(self, job: Job, plotter: PlotMetrics):
        self.job = job
        self.plotter = plotter
        self.train_test = np.concatenate(
            [np.zeros(job.n_logit_train_examples), np.ones(job.n_logit_test_examples)],
            axis=0,
        )

    def plot_train_test(self, plt_fn, name, row_metrics, col_metrics, mask):
        self.plotter.plot_array(
            plt_fn,
            name,
            row_metrics=row_metrics,
            col_metrics=col_metrics,
            mask=mask,
            group=self.train_test,
            group_names=["Train", "Test"],
        )

    def scatter(self, name, row_metrics, col_metrics, mask=None):
        self.plot_train_test(
            self.plotter.plt_scatter, name, row_metrics, col_metrics, mask=mask
        )

    def pair_corr(self, name, row_metrics, col_metrics, mask=None):
        self.plot_train_test(
            self.plotter.plt_pair_corr, name, row_metrics, col_metrics, mask=mask
        )

    def self_corr(self, name, row_metrics, col_metrics, mask=None):
        self.plot_train_test(
            self.plotter.plt_self_corr, name, row_metrics, col_metrics, mask=mask
        )

    def jaccard(self, name, row_metrics, col_metrics, mask=None):
        self.plot_train_test(
            self.plotter.plt_jaccard_curve, name, row_metrics, col_metrics, mask=mask
        )

    def pair_jaccard(self, name, row_metrics, col_metrics, mask=None):
        self.plot_train_test(
            self.plotter.plt_pair_jaccard, name, row_metrics, col_metrics, mask=mask
        )

    def hist(self, name, metrics, mask=None):
        self.plot_train_test(self.plotter.plt_hist, name, None, metrics, mask=mask)

    def quantiles(self, metrics, mask=None):
        self.plot_train_test(self.plotter.plt_quantiles, "qq", metrics, None, mask=mask)

    def plot_pairwise(self, metric_groups, prefix, mask=None):
        keys = list(metric_groups.keys())
        for i in range(len(keys)):
            self.hist(f"{prefix}-{keys[i]}", metric_groups[keys[i]], mask=mask)
            for j in range(i, len(keys)):
                name = f"{prefix}-{keys[i]}_{keys[j]}"
                row_metrics = metric_groups[keys[i]]
                col_metrics = metric_groups[keys[j]]
                self.scatter(name, row_metrics, col_metrics, mask=mask)
                self.pair_corr(name, row_metrics, col_metrics, mask=mask)
                # if list(row_metrics.values())[0].shape[0] > 1:  # can't do self corr over single rep
                #     self.self_corr(name, row_metrics, col_metrics, mask=mask)
                #     self.pair_jaccard(name, row_metrics, col_metrics, mask=mask)
                self.jaccard(name, row_metrics, col_metrics, mask=mask)

    def rank(self, metric):
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

    def plot_metrics_by_sample(self, metrics, perturbation):
        name = perturbation.name
        metrics = Metrics.filter_metrics(metrics, name)

        """ Q: is it reasonable to rank examples by mean/median noise needed to perturb?
            for metric noise_first_forget
            A) scatter meanS/medianS to id
                look for close to diagonal and not too spread out
            B) scatter meanS to stdS
                if homogenous over means, low variance as a % of noise scale, then mean is representative
                this allows averaging out S (sample dimension)
        """
        meanS = Metrics.apply(
            metrics,
            lambda metric: np.mean(metric, axis=-2, keepdims=True),
            suffix="mu",
        )
        medianS = Metrics.apply(
            metrics,
            lambda metric: np.median(metric, axis=-2, keepdims=True),
            suffix="med",
        )
        stdS = Metrics.apply(
            metrics,
            lambda metric: np.std(metric, axis=-2, keepdims=True),
            suffix="sd",
        )
        self.plot_pairwise({"id": metrics, "muS": meanS}, perturbation.name)
        self.plot_pairwise({"id": metrics, "medS": medianS}, perturbation.name)
        self.plot_pairwise({"stdS": stdS, "muS": meanS}, perturbation.name)
        """ C) scatter meanS to medianS
                if these look diagonal, then median is equivalent to mean
        """
        self.plot_pairwise({"medS": medianS, "muS": meanS}, perturbation.name)
        """ D) scatter rankNofmeanS/medianS to meanS/medianSofrankN
                if these look the same, then order of applying rank/mean doesn't matter
            A, B, C, D confirm use of median over S for rank corr with other metrics
        """
        rankofmedians = Metrics.apply(
            metrics,
            lambda metric: self.rank(np.median(metric, axis=-2)),
            suffix="rkmed",
        )
        medianofranks = Metrics.apply(
            metrics,
            lambda metric: np.median(self.rank(metric), axis=-2),
            suffix="medrk",
        )
        self.plot_pairwise(
            {"medofrk": medianofranks, "rkofmed": rankofmedians}, perturbation.name
        )

    def plot_all(self, metrics):
        # plot quantiles for each metric
        self.quantiles(metrics)
        # choose 1 of mean or median
        medianR = Metrics.apply(
            metrics,
            lambda metric: np.median(metric, axis=-2, keepdims=True),
            suffix="med",
        )
        """
        Q: are any metrics related?
            A) scatter identity to identity all metrics
                if any show strong trend, they are similar
        Q: is the variation in each metric due to inits or the examples themselves?
            A) corr identity to identity
            B) scatter/corr medianR to identity/rankN
                taking med improves corr if it is due to examples themselves, reduces corr if it is due to inits
                compare A med B identity and B med A identity to tell whether metric A or B depends more on inits
            C) scatter/corr medianR to medianR all metrics
                this tells whether variance in metrics is due to inits (higher corr here, due to averaging out variance)
                or due to examples (reduced corr here, due to correlated variance which cannot be averaged out)
        """
        self.plot_pairwise({"id": metrics, "medR": medianR}, "all")
        """
        Q: is variance over inits correlated to any metrics?
            A) scatter/corr stdR to identity/rankN all metrics
            B) scatter/corr stdR to medianR all metrics
        """
        stdR = Metrics.apply(
            metrics, lambda metric: np.std(metric, axis=-2, keepdims=True), suffix="std"
        )
        self.plot_pairwise({"stdR": stdR, "medR": medianR}, "all")

    def plot_learned_before_cutoff(self, metrics, learned_before_iter_inclusive):
        mask = self.mask_learned(
            metrics["train-first_learn"], learned_before_iter_inclusive
        )
        n_masked = np.sum(mask)
        medianR = Metrics.apply(
            metrics, lambda metric: self.masked_median(metric, mask), suffix="med"
        )
        plot_prefix = f"maskto{str(learned_before_iter_inclusive)}_n{n_masked}"
        # scatter plots with late_mean_prob coloring
        # self.plotter.plot_array(
        #     self.plotter.plt_scatter,
        #     f"{plot_prefix}-ids_mean_prob",
        #     row_metrics=metrics, col_metrics=metrics,
        #     mask=mask, group=metrics["train-late_mean_prob"])
        # self.plotter.plot_array(
        #     self.plotter.plt_scatter,
        #     f"{plot_prefix}-meds_mean_prob",
        #     row_metrics=medianR, col_metrics=medianR,
        #     mask=mask, group=metrics["train-late_mean_prob"])
        # other plots
        self.plot_pairwise({"id": metrics, "medR": medianR}, plot_prefix, mask=mask)
        # plot how many reps each example is learned in
        # n_reps_learned = {"n_reps_learned": np.sum(mask, axis=-2, keepdims=True)}
        # self.hist(plot_prefix + "n_reps_learned", n_reps_learned, mask, mask=mask)

    def masked_median(self, metric: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # take median over R but mask each array first
        assert metric.shape == mask.shape, (metric.shape, mask.shape)
        medians = [
            np.median(example[rep_mask]) for example, rep_mask in zip(metric.T, mask.T)
        ]
        medians = np.array(medians).reshape(1, -1)
        # nan occurs if all replicates masked, set to 0 (won't be plotted due to being masked)
        medians[np.isnan(medians)] = 0
        return medians

    def mask_learned(self, first_learn, learned_before_iter_inclusive=-1):
        if learned_before_iter_inclusive < 0:
            return None
        mask = first_learn <= learned_before_iter_inclusive
        print(
            "Include examples learned before iter",
            learned_before_iter_inclusive,
            stats_str(mask),
            "out of",
            stats_str(first_learn),
        )
        return mask
