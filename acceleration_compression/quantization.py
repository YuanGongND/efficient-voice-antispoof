import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as quant


class NetworkQuantization(object):
    def __init__(self, model, quant_method='dynamic', config='x86'):
        self.model = model
        self.print_model_size(model, 'Original Model')
        self.quant_method = quant_method
        self.qconfig = quant.get_default_qconfig('fbgemm') if config == 'x86' else quant.get_default_qconfig('qnnpack')

    @staticmethod
    def print_model_size(model, message=''):
        torch.save(model.state_dict(), "temp.p")
        print(message, 'Size (MB):', os.path.getsize("temp.p") / 1e6)
        os.remove('temp.p')

    def quantization(self):
        if self.quant_method == 'dynamic':
            quant_model = quant.quantize_dynamic(self.model, {nn.Linear, nn.Conv2d, nn.Conv1d}, dtype=torch.qint8)
        else:
            # Post-Training Static Quantization
            quant_model = copy.deepcopy(self.model)
            quant_model.qconfig = self.qconfig
            quant.prepare(quant_model, inplace=True)
            quant.convert(quant_model, inplace=True)
        self.print_model_size(quant_model, 'Quantized Model')
        return quant_model
