import os
import time
import numpy as np
from matplotlib import pyplot as plt


class PlotWeights:
    def __init__(self, job, noise_subdir=""):
        print(f"Loading models...")
        self.job = job
        self.noise_subdir = noise_subdir
        if self.noise_subdir == "":
            ckpt_source = self._load_training_epochs()
        else:
            ckpt_source = self._load_noise_epochs(noise_subdir)
        # store all weight layers
        self.layers = {}
        self.epochs = []
        self.reps = []
        for rep, epoch, ckpt in ckpt_source:
            self.epochs.append(epoch)
            self.reps.append(rep)
            if rep not in self.layers:
                self.layers[rep] = {}
            if epoch not in self.layers[rep]:
                self.layers[rep][epoch] = {}
            start_time = time.perf_counter()
            state_dict = ckpt["model_state_dict"]
            for layer_name, layer in state_dict.items():
                self.layers[rep][epoch][layer_name] = layer
            print(
                f"m={rep}, ep={epoch}, p={len(state_dict)}, t={time.perf_counter() - start_time}"
            )
        self.epochs = sorted(set(self.epochs))
        self.reps = sorted(set(self.reps))
        print(f"Models loaded:\n\treplicates=\n{self.reps} \n\tepochs=\n{self.epochs}")

    def _load_noise_epochs(self, subdir):
        for ckpt, name in self.job.load_checkpoints_from_dir(
            os.path.join(self.job.save_path, subdir), to_cpu=True
        ):
            model, noise, epoch = [
                int(component.split("=")[-1]) for component in name.split("-")
            ]
            rep = model * self.job.n_replicates + noise
            yield rep, epoch, ckpt

    def _load_training_epochs(self):
        for epoch in range(self.job.n_epochs + 1):  # 0 is init
            for ckpt, rep_name in self.job.load_checkpoints_by_epoch(
                epoch, to_cpu=True
            ):
                rep = int(rep_name[len("model") :])  # convert 'model0' to 0
                yield rep, epoch, ckpt

    def retrieve_layers(self, replicates=[], epochs=[], name_contains=[]):
        start_time = time.perf_counter()
        if type(epochs) is int:
            epochs = [epochs]
        if type(replicates) is int:
            replicates = [replicates]
        if type(name_contains) is str:
            name_contains = [name_contains]
        if len(epochs) == 0:  # include all
            epochs = self.epochs
        if len(replicates) == 0:  # include all
            replicates = self.reps

        print(
            f"Filtering replicates={replicates}, epochs={epochs}, names={name_contains}..."
        )
        n_layers = 0
        retrieved_layers = {}
        for rep in replicates:
            for epoch in epochs:
                for layer_name, layer in self.layers[rep][epoch].items():
                    if len(name_contains) == 0 or any(
                        x in layer_name for x in name_contains
                    ):
                        if layer_name not in retrieved_layers:
                            retrieved_layers[layer_name] = []
                            retrieved_layers[layer_name].append(layer)
                            n_layers += 1
        print(f"Retrieved {n_layers} layers t={time.perf_counter() - start_time}")
        return retrieved_layers

    def plot_histograms(self, layer_dict, filename):
        # layer_dict is list of outputs from retrieve_layers()
        # which is a list of dicts (layer names) of lists (layers)
        rows = sorted(layer_dict[0].keys())  # organize plots by sorted name
        cols = [i for i in range(len(layer_dict))]
        print(f"Plotting {len(cols)}x{len(rows)} histograms...")

        layer_min, layer_max = 1e9, -1e9
        data = []
        for name in rows:
            start_time = time.perf_counter()
            # flatten layers into single array
            data.append([np.stack(layer_dict[i][name]).flatten() for i in cols])
            layer_min = min(layer_min, *[np.min(x) for x in data[-1]])
            layer_max = max(layer_max, *[np.max(x) for x in data[-1]])
            print(f"{name}, l={len(data[-1])}, t={time.perf_counter() - start_time}")

        # require at least 2 rows as otherwise matplotlib returns 1D array of axes
        fig, axes = plt.subplots(
            max(len(rows), 2),
            max(len(cols), 2),
            sharey=True,
            figsize=(3 * max(len(cols), 2), 3 * max(len(rows), 2)),
        )
        for i, (row, ax_row) in enumerate(zip(data, axes)):
            for j, (layer, ax) in enumerate(zip(row, ax_row)):
                ax.hist(layer, bins="rice", density=True, range=(layer_min, layer_max))
                if i == 0:
                    ax.set_title(cols[j])
                if j == 0:
                    ax.set_ylabel(rows[i])
        self.job.save_obj_to_subdir(plt, f"weight_hist_{self.noise_subdir}", filename)

    def hist_layers_by_init(self, name_contains=[]):
        last_epoch = self.epochs[-1]  # use last epoch
        if type(name_contains) is str:
            name_contains = [name_contains]
        data = [
            self.retrieve_layers(
                replicates=i, epochs=last_epoch, name_contains=name_contains
            )
            for i in self.reps
        ]
        self.plot_histograms(data, "hist_by_init_" + "-".join(name_contains) + ".png")

    def hist_layers_by_epoch(self, name_contains=[]):
        if type(name_contains) is str:
            name_contains = [name_contains]
        data = [
            self.retrieve_layers(epochs=i, name_contains=name_contains)
            for i in self.epochs
        ]
        self.plot_histograms(data, "hist_by_epoch_" + "-".join(name_contains) + ".png")

    def hist_layers(self, name_contains=[]):
        if type(name_contains) is str:
            name_contains = [name_contains]
        data = [self.retrieve_layers(name_contains=name_contains)]
        self.plot_histograms(data, "hist_layers_" + "-".join(name_contains) + ".png")

    def plot_all(self, *filters):
        for filter in filters:
            self.hist_layers_by_init(name_contains=filter)
            self.hist_layers_by_epoch(name_contains=filter)
            self.hist_layers(name_contains=filter)
