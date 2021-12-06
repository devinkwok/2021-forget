import time
import numpy as np
import torch
from forget.job import evaluate_one_batch


class PrunePerturbation:
    def __init__(self, job):
        self.prune_type = 'prune_magnitude'
        self.job = job

    @property
    def subdir(self):
        return f'logits_{self.prune_type}'

    @property
    def scales(self):
        # always start from 0 to see if example was learned in trained model
        return np.linspace(
            0.0,
            float(self.job.hparams["prune scale max"]),
            int(self.job.hparams["prune num points"]),
        )

    def prune_logits(self):
        # load dataset to CUDA
        examples, labels = zip(*self.job.get_eval_dataset())
        examples = torch.stack(examples, dim=0).cuda()
        labels = torch.tensor(labels).cuda()

        # load trained models and sample noise
        for i, (ckpt, _) in enumerate(self.job.load_checkpoints_by_epoch(-1)):
            model = self.job.get_model(state_dict=ckpt["model_state_dict"])
            logits, accuracies = [], []

            def prune_logit():
                mask = None
                for scale in self.scales:
                    start_time = time.perf_counter()
                    mask = prune_mask(model, scale, current_mask=mask)
                    apply_mask(model, mask)  # modifies model in place
                    output, accuracy = evaluate_one_batch(model, examples, labels)
                    accuracies.append(accuracy)
                    logits.append(output)
                    print(
                        f"\ts={scale}, a={accuracy}, t={time.perf_counter() - start_time}"
                    )
                return {
                    "type": self.prune_type,
                    "logit": torch.stack(logits, dim=0),
                    "scale": self.scales,
                    "accuracy": accuracies,
                }

            self.job.cached(prune_logit, self.subdir, f"logits-model{i}.pt")

# from open_lth, copied to avoid import issues
def prunable_layer_names(model):
    """A list of the names of Tensors of this model that are valid for pruning.

    By default, only the weights of convolutional and linear layers are prunable.
    """

    return [name + '.weight' for name, module in model.named_modules() if
            isinstance(module, torch.nn.modules.conv.Conv2d) or
            isinstance(module, torch.nn.modules.linear.Linear)]

# from open_lth, copied to avoid import issues
def prune_mask(model, fraction, current_mask=None):
    empty_mask = {name: np.ones(list(model.state_dict()[name].shape))
        for name in prunable_layer_names(model)}
    if current_mask is None:
        current_mask = empty_mask

    # Determine the number of weights that need to be pruned.
    # MODIFICATION: always set fraction relative to original number of weights
    number_of_original_weights = np.sum([np.sum(v) for v in empty_mask.values()])
    number_of_remaining_weights = np.sum([np.sum(v) for v in current_mask.values()])
    already_pruned = number_of_original_weights - number_of_remaining_weights
    number_of_weights_to_prune = np.ceil(
        fraction * number_of_original_weights).astype(int) - already_pruned.astype(int)
    assert fraction > 0 and number_of_weights_to_prune > 0 or fraction == 0

    # Determine which layers can be pruned.
    prunable_tensors = set(prunable_layer_names(model))

    # Get the model weights.
    weights = {k: v.clone().cpu().detach().numpy()
                for k, v in model.state_dict().items()
                if k in prunable_tensors}

    # Create a vector of all the unpruned weights in the model.
    weight_vector = np.concatenate([v[current_mask[k] == 1] for k, v in weights.items()])
    threshold = np.sort(np.abs(weight_vector))[number_of_weights_to_prune]

    new_mask = {k: np.where(np.abs(v) > threshold, current_mask[k], np.zeros_like(v))
                        for k, v in weights.items()}
    for k in current_mask:
        if k not in new_mask:
            new_mask[k] = current_mask[k]

    return new_mask

# from open_lth, copied to avoid import issues
def apply_mask(model, mask):
    for name, param in model.named_parameters():
        if name in mask:
            param.data *= torch.tensor(mask[name]).cuda()
