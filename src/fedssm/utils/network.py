import torch
import copy
from typing import Tuple, Union


def scale_model_weights(weights, scaling_factor):
    """
        function for scaling a models weights
    """
    with torch.no_grad():
        for k in weights.keys():
            weights[k] = torch.mul(weights[k], scaling_factor)
    return weights


def sum_scaled_weights(scaled_weight_list, norm_factor=1):
    """
        Return the sum of the listed scaled weights.
        The is equivalent to a scaled average of the weights
    """
    w_avg = copy.deepcopy(scaled_weight_list[0])
    with torch.no_grad():
        for k in w_avg.keys():
            for i in range(1, len(scaled_weight_list)):
                w_avg[k] += scaled_weight_list[i][k]
            w_avg[k] /= norm_factor
    return w_avg


def apply_sparsity(
    topk, weight: torch.Tensor, return_scale_factors=False, *, method
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if method == "topk":
        return apply_topk(
            topk, weight, return_scale_factors=return_scale_factors
        )
    else:
        raise NotImplementedError(f"Invalid sparsity method {method}")


def apply_topk(topk: float, weight: torch.Tensor, return_scale_factors=False):
    """
    Given a weight tensor, retains the top self.current_topk of the weights and
    multiplies the rest by 0
    Inputs:
    - weight: A weight tensor, e.g., self.weight
    """
    # Retain only the topk weights, multiplying the rest by 0.
    frac_to_zero = 1 - topk
    with torch.no_grad():
        flat_weight = weight.flatten()
        # Want to convert it away from a special tensor, hence the float() call.
        _, idx = flat_weight.float().abs().sort()
        # @idx is a @special_tensors._SpecialTensor, but we need to convert it
        # to a normal tensor for indexing to work properly.
        idx = torch.tensor(idx, requires_grad=False)
        f = int(frac_to_zero * weight.numel())
        scale_factors = torch.ones_like(flat_weight, requires_grad=False)
        scale_factors[idx[:f]] = 0
        scale_factors = scale_factors.view_as(weight)

    ret = weight * scale_factors

    if return_scale_factors:
        return ret, scale_factors

    return ret
