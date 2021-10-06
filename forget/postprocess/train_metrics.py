import os
import time
import torch
import numpy as np
from transforms import *
from forget.postprocess.plot_metrics import PlotMetrics

class GenerateMetrics():

    def __init__(self, job, force_generate=False):
        # filter batch by train/test examples
        self.job = job
        self.force_generate = force_generate
        self.iter_mask = mask_iter_by_batch(self.job.hparams['batch size'],
                        0, int(self.job.hparams['eval number of train examples']))
        self.labels = np.array([y for _, y in self.job.get_eval_dataset()])

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
            outputs = [self._transform(i, source, transform_fn)
                        for i, source in enumerate(source_iterable)]
            torch.save(np.stack(outputs, axis=0), out_file)
        return torch.load(out_file, map_location=torch.device('cpu'))

    def replicate_by_epoch(self, model_dir, prefix='eval_logits'):  # source_iter
        # logits are saved after epochs, hence index from 1
        for i in range(1, self.job.n_epochs):
            file = os.path.join(model_dir, f'{prefix}={i}.pt')
            outputs = torch.load(file, map_location=torch.device('cpu'))
            yield file, outputs

    def noise_by_replicate():  # source_iter
        pass #TODO

    def batch_forgetting(self, output_prob_by_iter):  # as implemented by Toneva, same as Nikhil'ss
        return forgetting_events(output_prob_by_iter, batch_mask=self.batch_mask)

    def _train_eval_filter(self, ndarray, split_type):  # applies to metrics with N as last dim
        split_point = int(self.job.hparams['eval number of train examples'])
        if split_type == 'train':
            return ndarray[..., :split_point]
        elif self.split_type == 'test':
            return ndarray[..., split_point:]
        else:
            return ndarray

    def signed_prob(self, x):
        return signed_prob(x, self.labels)

    def gen_train_metrics_by_epoch(self):
        plotter = PlotMetrics(self.job)
        # R * E * (I x N x C) (list of R iterators over E)
        # R is replicates, E is epochs, I iters, N examples, C classes
        print("Loading signed probabilities...")
        softmaxes = [self.transform_inplace(self.replicate_by_epoch(r),
                        softmax) for r, _ in self.job.replicate_dirs()]
        # collate over E and transform over C to get R * (E x I x N)
        s_prob = [self.transform_collate(s, f'sgd_rep{i}',
                self.signed_prob) for i, s in enumerate(softmaxes)]
        s_prob = np.stack(s_prob, axis=0)  # stack to (R x E x I x N)
        n_epoch, n_iter = s_prob.shape[-3], s_prob.shape[-2]
        print("Loading metrics...")
        # each metric summarizes over I for (R x E x N)
        metrics_by_epoch = {
            'sgd_mean_prob': self.transform_collate('sgd', s_prob, mean_prob),
            'sgd_diff_norm': self.transform_collate('sgd', s_prob, diff_norm),
            # 'sgd_forgetting': self.generate_metrics(s_prob, forgetting_events),
            # 'sgd_batch_forgetting': self.generate_metrics(s_prob, self.batch_forgetting),
        }
        print("Plotting...")
        # plot by train/test/all
        for include in ['all', 'train', 'test']:
            name = f'-{include}'
            # label distribution
            plotter.plot_class_counts(name,
                    self._train_eval_filter(self.labels, include))
            # plot curves for selected examples
            curves = s_prob[..., np.arange(0, 10000, 1000)]
            # reshape to (R x EI x N)
            curves = curves.reshape(-1, n_epoch * n_iter, curves.shape[-1])
            # transpose to (N x R x EI)
            curves = np.transpose(curves, axes=(2, 0, 1))
            plotter.plot_score_curves('sgd' + name, curves)
            # plot mean over epochs (R x N)
            plotter.plot_metrics(metrics_by_epoch, name,
                filter=lambda x: np.mean(self._train_eval_filter(x, include), axis=1))
            # plot by epoch
            for i in range(n_epoch):
                plotter.plot_metrics(metrics_by_epoch, f'{name}-ep{i}',
                    filter=lambda x: self._train_eval_filter(x[:, i, :], include))
