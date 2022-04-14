import numpy as np
import torch
from forget.metrics import transforms
from forget.metrics.metrics import Metrics


class EvalMetrics(Metrics):
    def __init__(self, job, plotter):
        super().__init__(job, plotter)
        self.loss = torch.nn.CrossEntropyLoss(
            reduction="none"
        )  # keep per-example values
        _, labels = zip(*job.get_eval_dataset())
        self.labels = np.array(labels)
        self.true_output = np.zeros([len(self.labels), job.n_unique_labels])
        np.put_along_axis(self.true_output, self.labels.reshape(-1, 1), 1.0, axis=-1)

    def gen_metrics(self):
        def accuracy(logits):
            return (np.argmax(logits, axis=-1) == self.labels).reshape(1, -1)

        def margin(logits):
            error = transforms.softmax(torch.tensor(logits)) - self.true_output
            return np.linalg.norm(error, ord=2, axis=-1).reshape(1, -1)

        def loss(logits):
            with torch.no_grad():
                loss = self.loss(torch.tensor(logits), torch.tensor(self.labels))
            return loss.numpy().reshape(1, -1)

        def get_output_vector_fn(component):
            # return lambda fn to capture component in closure
            return lambda x: x[..., component].reshape(1, -1)

        metric_generators = {
            "accuracy": accuracy,
            "margin": margin,
            "loss": loss,
        }
        # for i in range(self.job.n_unique_labels):
        #     metric_generators[f"logit_class{i}"] = get_output_vector_fn(i)

        self._gen_metrics(
            "eval", self.eval_logits(), metric_generators, do_plot_curves=False
        )

    def eval_logits(self):
        for i, _ in enumerate(self.job.replicates()):
            logits = self.job.load_from_replicate(
                i, self.job.n_epochs, "logits-ep", to_cpu=True
            )[-1]
            # get logits only from very last iteration (I x N x C) -> (N x C)
            assert logits.shape == (
                self.job.n_eval_examples,
                self.job.n_unique_labels,
            ), logits.shape
            yield logits.numpy()
