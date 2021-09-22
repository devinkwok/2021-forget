import os
import time
import numpy as np
from itertools import chain
from matplotlib import pyplot as plt

class PlotWeights():

    def __init__(self, job):
        start_time = time.perf_counter()
        self.job = job
        self.layers = {}
        for epoch in range(job.n_epochs):
            self.layers[epoch] = {}
            for rep, ckpt in enumerate(job.load_checkpoints(epoch, to_cpu=True)):
                self.layers[epoch][rep] = {}
                state_dict = ckpt['model_state_dict']
                print(f'Loading model{rep} epoch={epoch}.pt with {len(state_dict)} params...')
                for layer_name, layer in state_dict.items():
                    self.layers[epoch][rep][layer_name] = layer
        print(f'Models loaded t={time.perf_counter - start_time}')

    def retrieve_layers(self, replicates=[], epochs=[], name_contains=[]):
        start_time = time.perf_counter()
        if type(epochs) is int:
            epochs = [epochs]
        if type(replicates) is int:
            replicates = [replicates]
        if type(name_contains) is str:
            name_contains = [name_contains]
        if len(epochs) == 0:  # include all
            epochs = np.arange(self.job.n_epochs)
        if len(replicates) == 0:  # include all
            replicates = np.arange(self.job.n_replicates)

        print(f'Filtering replicates={replicates}, epochs={epochs}, names={name_contains}...')
        n_layers = 0
        retrieved_layers = {}
        for rep in replicates:
            for epoch in epochs:
                for layer_name, layer in self.layers[rep][epoch].items():
                    if len(name_contains) == 0 or \
                            any(x in layer_name for x in name_contains):
                        if layer_name not in retrieved_layers:
                            retrieved_layers[layer_name] = []
                            retrieved_layers[layer_name].append(layer)
                            n_layers += 1
        print(f'Retrieved {n_layers} layers t={time.perf_counter - start_time}')
        return retrieved_layers

    def plot_histograms(self, data, filename, n_bins=10):
        print(f'Plotting {len(x_labels)}x{len(y_labels)} histograms...')
        start_time = time.perf_counter()
        # data is list of outputs from retrieve_layers()
        # which is a list of dicts (layer names) of lists (layers)
        x_labels = sorted(data[0].keys())  # organize plots by sorted name
        y_labels = [i for i in range(len(data))]
        layer_min, layer_max = -1e9, 1e9
        for name in x_labels:
            # flatten layers into single array
            data.append([np.flatten(np.stack(data[i][name]))
                        for i in y_labels])
            print(f'Flattened {name} with {len(data[-1])} entries...')
            layer_min = min(layer_min, *[np.min(x) for x in data[-1]])
            layer_max = max(layer_max, *[np.max(x) for x in data[-1]])
        fig, axes = plt.subplots(len(x_labels), len(y_labels),
                                sharey=True, tight_layout=True)
        for i, x in enumerate(chain(*data)):
            axes[i].hist(x, bins=n_bins, density=True, range=(layer_min, layer_max))
            if i < len(x_labels):
                axes[i].set_title(x_labels[i])
            if i % len(y_labels) == 0:
                axes[i].set_ylabel(y_labels[len(data) // i])
        fig_path = os.path.join(self.job.save_path, 'weight_histograms', filename)
        print(f'Saving to {fig_path} t={time.perf_counter - start_time}.')
        plt.savefig(fig_path)

    def hist_layers_by_init(self, name_contains=[]):
        last_epoch = self.job.n_epochs - 1  # use last epoch
        data = [self.retrieve_layers(replicates=i,
                    epochs=last_epoch, name_contains=name_contains)
                for i in range(self.job.n_replicates)]
        self.plot_histograms(data, 'hist_by_init' + '-'.join(name_contains) + '.png')

    def hist_layers_by_epoch(self, name_contains=[]):
        data = [self.retrieve_layers(epochs=i, name_contains=name_contains)
            for i in range(self.job.n_epochs)]
        self.plot_histograms(data, 'hist_by_epoch' + '-'.join(name_contains) + '.png')

    def hist_layers(self, name_contains=[]):
        data = [self.retrieve_layers(name_contains=name_contains)]
        self.plot_histograms(data, 'hist_layers' + '-'.join(name_contains) + '.png')
