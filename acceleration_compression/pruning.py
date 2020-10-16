import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune


class NetworkPruning(object):
    r""" Basic Network Pruning Class based on torch.nn.utils.prune
    """
    def __init__(self, model, prune_method='L1Unstructured', percentage=0.2, model_copy=False):
        self.model = model
        self.model_copy = model_copy
        if self.model_copy:
            self.original_model = copy.deepcopy(model)
        self.prune_method = prune_method.lower()
        self.percentage = percentage

    def get_original_model(self):
        if self.model_copy:
            return self.original_model
        else:
            return None

    def get_prune_parameters(self):
        parameters = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                parameters.append((module, 'weight'))
                if module.bias is not None:
                    parameters.append((module, 'bias'))
        return parameters

    def get_prune_method(self):
        if 'l1' in self.prune_method:
            return prune.L1Unstructured
        elif 'ln' in self.prune_method:
            return prune.LnStructured
        elif 'random' in self.prune_method:
            if 'unstructured' not in self.prune_method:
                return prune.RandomStructured
            else:
                return prune.RandomUnstructured

    @staticmethod
    def print_size_of_model(model):
        torch.save(model.state_dict(), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p") / 1e6)
        os.remove('temp.p')

    @staticmethod
    def get_num_parameters(model, is_nonzero=False):
        count = 0
        for p in model.parameters():
            if is_nonzero:
                count += torch.nonzero(p, as_tuple=False).shape[0]
            else:
                count += p.numel()
        return count

    def pruning(self):
        parameters_to_prune = self.get_prune_parameters()
        prune_method = self.get_prune_method()
        prune.global_unstructured(
            parameters=parameters_to_prune,
            pruning_method=prune_method,
            amount=self.percentage,
        )

        # clean up re-parameterization
        for module, name in parameters_to_prune:
            prune.remove(module, name)

        return self.model
