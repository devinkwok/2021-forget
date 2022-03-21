import numpy as np
import torch
from forget.perturb.perturb import Perturbation


class PrunePerturbation(Perturbation):
    def __init__(self, job):
        super().__init__("prune_magn_notrain", job, False)

    def apply_perturbation(self, noise, scale_idx, scale, model, examples):
        mask = prune_mask(model, scale)
        # copy model to avoid affecting original
        pruned_model = self.job.get_model(model.state_dict())
        apply_mask(pruned_model, mask)  # modifies model in place
        return pruned_model, examples

    def gen_noise_sample(self):
        return None  # not used


# from open_lth, copied to avoid import issues
def prunable_layer_names(model):
    """A list of the names of Tensors of this model that are valid for pruning.

    By default, only the weights of convolutional and linear layers are prunable.
    """

    return [
        name + ".weight"
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.modules.conv.Conv2d)
        or isinstance(module, torch.nn.modules.linear.Linear)
    ]


# from open_lth, copied to avoid import issues
def prune_mask(model, fraction):
    current_mask = {
        name: np.ones(list(model.state_dict()[name].shape))
        for name in prunable_layer_names(model)
    }

    # Determine the number of weights that need to be pruned.
    # MODIFICATION: current_mask is always None
    number_of_original_weights = np.sum([np.sum(v) for v in current_mask.values()])
    number_of_weights_to_prune = np.ceil(fraction * number_of_original_weights).astype(
        int
    )
    assert fraction > 0 and number_of_weights_to_prune > 0 or fraction == 0

    # Determine which layers can be pruned.
    prunable_tensors = set(prunable_layer_names(model))

    # Get the model weights.
    weights = {
        k: v.clone().cpu().detach().numpy()
        for k, v in model.state_dict().items()
        if k in prunable_tensors
    }

    # Create a vector of all the unpruned weights in the model.
    weight_vector = np.concatenate(
        [v[current_mask[k] == 1] for k, v in weights.items()]
    )
    threshold = np.sort(np.abs(weight_vector))[number_of_weights_to_prune]

    new_mask = {
        k: np.where(np.abs(v) > threshold, current_mask[k], np.zeros_like(v))
        for k, v in weights.items()
    }
    for k in current_mask:
        if k not in new_mask:
            new_mask[k] = current_mask[k]

    return new_mask


# from open_lth, copied to avoid import issues
def apply_mask(model, mask):
    for name, param in model.named_parameters():
        if name in mask:
            param.data *= torch.tensor(mask[name]).cuda()
