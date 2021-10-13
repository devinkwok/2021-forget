import os
import time
import torch
import numpy as np
from forget.postprocess.transforms import *
from forget.damage.noise import noise_scales
from forget.postprocess.plot_metrics import PlotMetrics

class GenerateMetrics():

    def __init__(self, job, force_generate=False):
        # filter batch by train/test examples
        self.job = job
        self.force_generate = force_generate
        eval_dataset = self.job.get_eval_dataset()
        self.labels = np.array([y for _, y in eval_dataset])
        self.iter_mask = mask_iter_by_batch(int(self.job.hparams['batch size']),
            len(self.job.get_train_dataset()), 0, len(self.labels))
        scale = noise_scales(self.job)
        last_scale_item = [scale[-1] + scale[0]]
        self.scale = np.concatenate([scale, last_scale_item])

    def _transform(self, name, source, transform_fn):
        start_time = time.perf_counter()
        output = transform_fn(source)
        print(f'\t{stats_str(output)} t={time.perf_counter() - start_time} {name}')
        return output

    def transform_inplace(self, source_file_iterable, transform_fn):
        for source_file, source in source_file_iterable:
            source_path, ext = os.path.splitext(source_file)
            out_file = source_path + '-' + transform_fn.__name__ + ext
            if not os.path.exists(out_file) or self.force_generate:
                print(f'Generating in-place {transform_fn.__name__} ' + \
                        f'from {source_file_iterable.__name__}:')
                output = self._transform(source_file, source, transform_fn)
                torch.save(output, out_file)
            yield torch.load(out_file, map_location=torch.device('cpu'))

    def transform_collate(self, name, source_iterable, transform_fn):
        fname = f'{name}-{transform_fn.__name__}.metric'
        out_file = os.path.join(self.job.save_path, 'metrics', fname)
        # check if output of transform_fn already exists for source_iterable
        if not os.path.exists(out_file) or self.force_generate:
            print(f'Generating collated {transform_fn.__name__} ' + \
                    f'from {name}:')
            outputs = [self._transform(str(i), source, transform_fn)
                        for i, source in enumerate(source_iterable)]
            torch.save(np.stack(outputs, axis=0), out_file)
        return torch.load(out_file, map_location=torch.device('cpu'))

    def replicate_by_epoch(self, model_dir, prefix='eval_logits'):  # source_iter
        # logits are saved after epochs, hence index from 1
        for i in range(1, self.job.n_epochs):
            file = os.path.join(model_dir, f'{prefix}={i}.pt')
            outputs = torch.load(file, map_location=torch.device('cpu'))
            yield file, outputs

    def noise_by_sample(self, noise_dir, replicate_name):  # source_iter
        for i in range(int(self.job.hparams['num noise samples'])):
            file = os.path.join(noise_dir, f'logits-{replicate_name}-noise{i}.pt')
            outputs = torch.load(file, map_location=torch.device('cpu'))
            yield file, outputs['logit']

    def batch_forgetting(self, output_prob_by_iter):  # as implemented by Toneva, same as Nikhil'ss
        return forgetting_events(output_prob_by_iter, iter_mask=self.iter_mask)

    def _train_eval_filter(self, ndarray, split_type):  # applies to metrics with N as last dim
        split_point = int(self.job.hparams['eval number of train examples'])
        if split_type == 'train':
            return ndarray[..., :split_point]
        elif split_type == 'test':
            return ndarray[..., split_point:]
        else:
            return ndarray

    def signed_prob(self, x):
        return signed_prob(x, self.labels)

    def first_forget(self, x):
        return first_forget(x, self.scale)

    def gen_train_metrics_by_epoch(self):
        plotter = PlotMetrics(self.job)
        # R * E * (I x N x C) (list of R iterators over E)
        # R is replicates, E is epochs, I iters, N examples, C classes
        print("Loading signed probabilities...")
        softmaxes = [self.transform_inplace(self.replicate_by_epoch(r),
                        softmax) for r, _ in self.job.replicates()]
        # collate over E and transform over C to get R * (E x I x N)
        s_prob = [self.transform_collate(f'sgd_rep{i}', s,
                self.signed_prob) for i, s in enumerate(softmaxes)]
        s_prob = np.stack(s_prob, axis=0)  # stack to (R x E x I x N)
        n_epoch, n_iter, n_example = s_prob.shape[-3], s_prob.shape[-2], s_prob.shape[-1]
        print("Loading metrics...")
        # batch forgetting can't be calculated per epoch, change to (R x 1 x EI x N)
        s_prob_no_epoch = s_prob.reshape(-1, 1, n_epoch * n_iter, n_example)
        # summarizes over I for (R x E x N)
        metrics = {
            'sgd_mean_prob': self.transform_collate('sgd', s_prob, mean_prob),
            'sgd_diff_norm': self.transform_collate('sgd', s_prob, diff_norm),
            'sgd_forget': self.transform_collate('sgd', s_prob_no_epoch, forgetting_events),
            'sgd_batch_forget': self.transform_collate('sgd', s_prob_no_epoch, self.batch_forgetting),
        }
        metrics_by_epoch = {
            'sgd_mean_prob': self.transform_collate('sgd', s_prob, mean_prob),
            'sgd_diff_norm': self.transform_collate('sgd', s_prob, diff_norm),
            'sgd_forget': self.transform_collate('sgd_per_ep', s_prob, forgetting_events),
        }
        print("Plotting...")
        # plot by train/test/all
        for include in ['all', 'train', 'test']:
            name = f'-{include}'
            # label distribution
            plotter.plot_class_counts(name,
                    self._train_eval_filter(self.labels, include))
            # probability curves
            filtered_prob = self._train_eval_filter(s_prob_no_epoch, include).squeeze(axis=1)
            mean_over_epochs = {f'{k}{name}': np.mean(
                self._train_eval_filter(v, include), axis=1) \
                for k, v in metrics.items()}
            plotter.plot_curves_by_rank(filtered_prob, mean_over_epochs)
            # plot metric over epochs (R x N)
            plotter.plot_metric_rank_qq(mean_over_epochs)
            plotter.plot_metric_scatter_array(name, mean_over_epochs)
            plotter.plot_metric_rank_corr_array(name, mean_over_epochs)
            # plot metrics per epoch
            for i in range(n_epoch):
                by_epoch = {f'{k}{name}-ep{i}': self._train_eval_filter(
                            v[:, i, :], include) \
                            for k, v in metrics_by_epoch.items()}
                plotter.plot_metric_scatter_array(f'{name}-ep{i}', by_epoch)
                plotter.plot_metric_rank_corr_array(f'{name}-ep{i}', by_epoch)
        return metrics, metrics_by_epoch

    def gen_noise_metrics(self, noise_type, name_contains):
        plotter = PlotMetrics(self.job)
        # R * S * (I x N x C) (list of R iterators over S)
        # R is replicates, S is noise samples, I iters, N examples, C classes
        print("Loading signed probabilities...")
        noise_dir = os.path.join(self.job.save_path,
                    f'logits_noise_{noise_type}_{"-".join(name_contains)}')
        softmaxes = [self.transform_inplace(self.noise_by_sample(noise_dir, r),
                        softmax) for _, r in self.job.replicates()]
        # collate over S and transform over C to get R * (S x I x N)
        s_prob = [self.transform_collate(f'noise_sample{i}', s,
                self.signed_prob) for i, s in enumerate(softmaxes)]
        s_prob = np.stack(s_prob, axis=0)  # stack to (R x S x I x N)
        print("Loading metrics...")
        # summarizes over I for (R x S x N)
        metrics = {
            'noise_mean_prob': self.transform_collate('noise', s_prob, mean_prob),
            'noise_diff_norm': self.transform_collate('noise', s_prob, diff_norm),
            'first_forget': self.transform_collate('noise', s_prob, self.first_forget),
        }
        print("Plotting...")
        # plot by train/test/all
        for include in ['all', 'train', 'test']:
            filtered_prob = self._train_eval_filter(s_prob, include)
            metrics_by_noise = {f'{k}-{include}': self._train_eval_filter(v, include) \
                                for k, v in metrics.items()}
            plotter.plot_curves_by_rank(filtered_prob, metrics_by_noise)
            # plot (R, S, N), mean over S (noises)
            plotter.plot_metric_scatter_array(f'-sample-{include}', metrics_by_noise)
            plotter.plot_metric_rank_corr_array(f'-sample-{include}', metrics_by_noise)
            # plot (S, R, N), mean over R (inits)
            metrics_by_rep = {k: v.transpose(1, 0, 2) for k, v in metrics_by_noise.items()}
            plotter.plot_metric_scatter_array(f'-init-{include}', metrics_by_rep)
            plotter.plot_metric_rank_corr_array(f'-init-{include}', metrics_by_rep)
        return metrics

    def gen_train_to_noise_metrics(self, train_metrics, train_metrics_by_epoch, noise_metrics):
        plotter = PlotMetrics(self.job)
        for include in ['all', 'train', 'test']:
            metrics_by_noise = {f'{k}-{include}': self._train_eval_filter(v, include) \
                                for k, v in noise_metrics.items()}
            # plot (R, S, N), mean over S (noises)
            metrics = {**train_metrics, **metrics_by_noise}
            plotter.plot_metric_scatter_array(f'-sample-{include}', metrics)
            plotter.plot_metric_rank_corr_array(f'-sample-{include}', metrics)
            # plot (S, R, N), mean over R (inits)
            metrics_by_rep = {k: v.transpose(1, 0, 2) for k, v in metrics_by_noise.items()}
            metrics = {**train_metrics, **metrics_by_rep}
            plotter.plot_metric_scatter_array(f'-init-{include}', metrics)
            plotter.plot_metric_rank_corr_array(f'-init-{include}', metrics)
            for i in range(self.job.hparams['num epochs']):
                by_epoch = {f'{k}-{include}-ep{i}': self._train_eval_filter(
                            v[:, i, :], include) \
                            for k, v in train_metrics_by_epoch.items()}
                # plot (R, S, N), mean over S (noises)
                metrics = {**by_epoch, **metrics_by_noise}
                plotter.plot_metric_scatter_array(f'-{include}-ep{i}', metrics)
                plotter.plot_metric_rank_corr_array(f'-{include}-ep{i}', metrics)
                # plot (S, R, N), mean over R (inits)
                metrics = {**by_epoch, **metrics_by_rep}
                plotter.plot_metric_scatter_array(f'-{include}-ep{i}', metrics)
                plotter.plot_metric_rank_corr_array(f'-{include}-ep{i}', metrics)
