import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as quant


class NetworkQuantization(object):
    def __init__(self, model, quant_method='dynamic', config='x86', calibration_loader=None):
        '''
        :param config: platform switch
        :type config: x86, pi, jetson
        '''
        self.model = model
        self.print_model_size(model, 'Original Model')
        self.quant_method = quant_method
        self.config = config
        self.qconfig = quant.get_default_qconfig('fbgemm') if config == 'x86' else quant.get_default_qconfig('qnnpack')

        # For post training static quantization calibration, typically the training data loader
        self.calibration_loader = calibration_loader
        assert self.quant_method == 'static' and self.calibration_loader is not None, \
            'Post training static quantization requires calibration loader (training loader)!'

    @staticmethod
    def print_model_size(model, message=''):
        torch.save(model.state_dict(), "temp.p")
        print(message, 'Size (MB):', os.path.getsize("temp.p") / 1e6)
        size = os.path.getsize("temp.p") / 1e6
        os.remove('temp.p')
        return size

    @staticmethod
    def get_num_parameters(model, is_nonzero=False):
        count = 0
        for p in model.parameters():
            if is_nonzero:
                count += torch.nonzero(p, as_tuple=False).shape[0]
            else:
                count += p.numel()
        return count

    def quantization(self):
        if self.quant_method == 'dynamic':
            torch.backends.quantized.engine = 'fbgemm' if self.config == 'x86' else 'qnnpack'
            quant_model = quant.quantize_dynamic(self.model, {nn.Linear, nn.Conv2d, nn.Conv1d}, dtype=torch.qint8)
        else:
            # Post-Training Static Quantization
            quant_model = copy.deepcopy(self.model)
            quant_model.eval()
            quant_model.fuse_model()
            quant_model.qconfig = self.qconfig
            quant.prepare(quant_model, inplace=True)
            self.calibrate_model(quant_model, self.calibration_loader)
            quant.convert(quant_model, inplace=True)
        self.print_model_size(quant_model, 'Quantized Model')
        return quant_model

    @staticmethod
    def calibrate_model(model, loader, calibrate_batches=20, device=torch.device('cpu:0')):
        model.to(device)
        model.eval()
        cnt = 0
        with torch.no_grad():
            for _, inputs, _ in loader:
                inputs = inputs.to(device)
                _ = model(inputs)
                cnt += 1
                if cnt >= calibrate_batches:
                    break
