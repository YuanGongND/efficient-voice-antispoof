from __future__ import division

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tensorly as tl
from scipy.optimize import minimize_scalar
from tensorly.decomposition import parafac, partial_tucker


tl.set_backend('pytorch')


class NetworkDecomposition(object):
    """
    This class is heavily inspired and adapted from the following projects:
        https://github.com/larry0123du/Decompose-CNN/
        https://github.com/jacobgil/pytorch-tensor-decompositions/
    """
    def __init__(self, model, rank=0, decomp_type='tucker', model_copy=False):
        super(NetworkDecomposition, self).__init__()
        self.model = model
        self.model_copy = model_copy
        if self.model_copy:
            self.original_model = copy.deepcopy(model)
        self.decomp_type = decomp_type
        self.rank = rank

    @staticmethod
    def print_size_of_model(model):
        torch.save(model.state_dict(), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p") / 1e6)
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

    def estimate_rank(self, W):
        if self.rank > 0:
            if self.decomp_type == 'tucker':
                return [self.rank, self.rank]
            elif self.decomp_type == 'cp':
                return self.rank

        mode3 = tl.base.unfold(W, 0)
        mode4 = tl.base.unfold(W, 1)
        mode4 = mode4.t() if mode4.shape[0] > mode4.shape[1] else mode4
        diag0 = EVBMF(mode3)
        diag1 = EVBMF(mode4)
        if len(diag0) == 0 or len(diag1) == 0:
            rank = max(W.shape[0] // 4, W.shape[1] // 4)
            if self.decomp_type == 'tucker':
                return [rank, rank]
            elif self.decomp_type == 'cp':
                return rank
        d1 = diag0.shape[0]
        d2 = diag1.shape[1]

        del mode3
        del mode4
        del diag0
        del diag1

        # round to multiples of 16
        if self.decomp_type == 'tucker':
            return [int(np.ceil(d1 / 16) * 16), int(np.ceil(d2 / 16) * 16)]
        elif self.decomp_type == 'cp':
            return int(np.ceil(max(d1, d2) / 16) * 16)

    def conv2d_decomp(self, layer):
        W = layer.weight.data
        ranks = self.estimate_rank(W)
        print('Ranks:', ranks)
        if ranks == 0:
            return layer
        last, first, vertical, horizontal = [None for _ in range(4)]
        if self.decomp_type == 'tucker':
            core, [last, first] = partial_tucker(W, modes=[0, 1], ranks=ranks, init='svd')
        elif self.decomp_type == 'cp':
            weights, [last, first, vertical, horizontal] = parafac(W, rank=ranks, init='random')

        pointwise_s_to_r_layer = nn.Conv2d(in_channels=first.shape[0],
                                           out_channels=first.shape[1],
                                           kernel_size=1,
                                           padding=0,
                                           bias=False)
        pointwise_s_to_r_layer.weight.data = first.t().unsqueeze(-1).unsqueeze(-1)

        core_layer = None
        if self.decomp_type == 'tucker':
            core_layer = nn.Conv2d(in_channels=core.shape[1],
                                   out_channels=core.shape[0],
                                   kernel_size=layer.kernel_size,
                                   stride=layer.stride,
                                   padding=layer.padding,
                                   dilation=layer.dilation,
                                   bias=False)
            core_layer.weight.data = core
        elif self.decomp_type == 'cp':
            # depthwise_r_to_t_layer
            core_layer = nn.Conv2d(in_channels=ranks,
                                   out_channels=ranks,
                                   kernel_size=vertical.shape[0],
                                   stride=layer.stride,
                                   padding=layer.padding,
                                   dilation=layer.dilation,
                                   groups=ranks,
                                   bias=False)
            core_layer.weight.data = torch.bmm(vertical.t().unsqueeze(-1), horizontal.t().unsqueeze(-2)).unsqueeze(1)

        pointwise_r_to_t_layer = nn.Conv2d(in_channels=last.shape[1],
                                           out_channels=last.shape[0],
                                           kernel_size=1,
                                           padding=0,
                                           bias=True)
        pointwise_r_to_t_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)

        if layer.bias is not None:
            pointwise_r_to_t_layer.bias.data = layer.bias.data

        new_layer = nn.Sequential(
            pointwise_s_to_r_layer,
            core_layer,
            pointwise_r_to_t_layer
        )

        return new_layer

    def linear_decomp(self, layer):
        W = layer.weight.data
        ranks = self.estimate_rank(W)
        print('Ranks:', ranks)
        if ranks == 0:
            return layer
        last, first = [None for _ in range(2)]
        if self.decomp_type == 'tucker':
            first, [last] = partial_tucker(W, modes=[0], ranks=ranks, init='svd')
            first_layer = nn.Linear(first.shape[1], first.shape[0], bias=False)
            first_layer.weight.data = first
            last_layer = nn.Linear(last.shape[1], last.shape[0], bias=True)
            last_layer.weight.data = last
        elif self.decomp_type == 'cp':
            weights, [last, first] = parafac(W, rank=ranks, init='random')
            first_layer = nn.Linear(first.shape[0], first.shape[1], bias=False)
            first_layer.weight.data = first.t()
            last_layer = nn.Linear(last.shape[1], last.shape[0], bias=True)
            last_layer.weight.data = last

        if layer.bias is not None:
            last_layer.bias.data = layer.bias.data

        new_layer = nn.Sequential(
            first_layer,
            last_layer
        )

        return new_layer

    def decomposition(self):
        for name, layer in self.model.named_modules():
            new_layer = None
            if isinstance(layer, nn.Conv2d):
                print('Decomposing:', name, end=' - ')
                new_layer = self.conv2d_decomp(layer)
            elif isinstance(layer, nn.Linear):
                print('Decomposing:', name, end=' - ')
                new_layer = self.linear_decomp(layer)
            else:
                continue

            # name in one of the formats: 'layer3.0.conv1.0', 'layer3.5.conv3', 'att9.0' or 'cnn1'
            tokens = name.split('.')
            if len(tokens) == 4:
                # ResNet18: layer3.0.conv1.0
                level1 = getattr(self.model, tokens[0])
                level2 = level1[int(tokens[1])]
                level3 = getattr(level2, tokens[2])
                level3[int(tokens[3])] = new_layer
                del level1
                del level2
                del level3
            elif len(tokens) == 3:
                # ResNet50: layer3.5.conv3
                level1 = getattr(self.model, tokens[0])
                level2 = level1[int(tokens[1])]
                setattr(level2, tokens[2], new_layer)
                del level1
                del level2
            elif len(tokens) == 2:
                level1 = getattr(self.model, tokens[0])
                level1[int(tokens[1])] = new_layer
            elif len(tokens) == 1:
                setattr(self.model, tokens[0], new_layer)

        return self.model


def EVBMF(Y, sigma2=None, H=None):
    """Implementation of the analytical solution to Empirical Variational Bayes Matrix Factorization.

    This function can be used to calculate the analytical solution to empirical VBMF.
    This is based on the paper and MatLab code by Nakajima et al.:
    "Global analytic solution of fully-observed variational Bayesian matrix factorization."

    Notes
    -----
        If sigma2 is unspecified, it is estimated by minimizing the free energy.
        If H is unspecified, it is set to the smallest of the sides of the input Y.

    Attributes
    ----------
    Y : numpy-array
        Input matrix that is to be factorized. Y has shape (L,M), where L<=M.

    sigma2 : int or None (default=None)
        Variance of the noise on Y.

    H : int or None (default = None)
        Maximum rank of the factorized matrices.

    Returns
    -------
    U : numpy-array
        Left-singular vectors.

    S : numpy-array
        Diagonal matrix of singular values.

    V : numpy-array
        Right-singular vectors.

    post : dictionary
        Dictionary containing the computed posterior values.


    References
    ----------
    .. [1] Nakajima, Shinichi, et al. "Global analytic solution of fully-observed variational Bayesian matrix factorization." Journal of Machine Learning Research 14.Jan (2013): 1-37.

    .. [2] Nakajima, Shinichi, et al. "Perfect dimensionality recovery by variational Bayesian PCA." Advances in Neural Information Processing Systems. 2012.
    """
    L, M = Y.shape  # has to be L<=M

    if H is None:
        H = L

    alpha = L / M
    tauubar = 2.5129 * np.sqrt(alpha)

    # SVD of the input matrix, max rank of H
    U, s, V = torch.svd(Y)
    U = U[:, :H]
    s = s[:H]
    V[:H].t_()

    # Calculate residual
    residual = 0.
    if H < L:
        residual = torch.sum(torch.sum(Y ** 2) - torch.sum(s ** 2))

    # Estimation of the variance when sigma2 is unspecified
    if sigma2 is None:
        xubar = (1 + tauubar) * (1 + alpha / tauubar)
        eH_ub = int(np.min([np.ceil(L / (1 + alpha)) - 1, H])) - 1
        upper_bound = (torch.sum(s ** 2) + residual) / (L * M)
        upper_bound = upper_bound.item()
        lower_bound = np.max([s[eH_ub + 1] ** 2 / (M * xubar), torch.mean(s[eH_ub + 1:] ** 2) / M])
        scale = 1.  # /lower_bound
        s = s * np.sqrt(scale)
        residual = residual * scale
        lower_bound = lower_bound * scale
        upper_bound = upper_bound * scale
        sigma2_opt = minimize_scalar(EVBsigma2, args=(L, M, s, residual, xubar), bounds=[lower_bound, upper_bound],
                                     method='Bounded')
        sigma2 = sigma2_opt.x

    # Threshold gamma term
    threshold = np.sqrt(M * sigma2 * (1 + tauubar) * (1 + alpha / tauubar))

    pos = torch.sum(s > threshold)
    if pos == 0:
        return np.array([])

    # Formula (15) from [2]
    d = torch.mul(s[:pos] / 2,
                  1 - (L + M) * sigma2 / s[:pos] ** 2 + torch.sqrt(
                      (1 - ((L + M) * sigma2) / s[:pos] ** 2) ** 2 - (4 * L * M * sigma2 ** 2) / s[:pos] ** 4)
                  )

    return torch.diag(d)


def EVBsigma2(sigma2, L, M, s, residual, xubar):
    H = len(s)

    alpha = L / M
    x = s ** 2 / (M * sigma2)

    z1 = x[x > xubar]
    z2 = x[x <= xubar]
    tau_z1 = tau(z1, alpha)

    term1 = torch.sum(z2 - torch.log(z2))
    term2 = torch.sum(z1 - tau_z1)
    term3 = torch.sum(torch.log((tau_z1 + 1) / z1))
    term4 = alpha * torch.sum(torch.log(tau_z1 / alpha + 1))

    obj = term1 + term2 + term3 + term4 + residual / (M * sigma2) + (L - H) * np.log(sigma2)

    return obj


def tau(x, alpha):
    return 0.5 * (x - (1 + alpha) + torch.sqrt((x - (1 + alpha)) ** 2 - 4 * alpha))


# def EVBMF(Y, sigma2=None, H=None):
#     """Implementation of the analytical solution to Empirical Variational Bayes Matrix Factorization.
#
#     This function can be used to calculate the analytical solution to empirical VBMF.
#     This is based on the paper and MatLab code by Nakajima et al.:
#     "Global analytic solution of fully-observed variational Bayesian matrix factorization."
#
#     Notes
#     -----
#         If sigma2 is unspecified, it is estimated by minimizing the free energy.
#         If H is unspecified, it is set to the smallest of the sides of the input Y.
#
#     Attributes
#     ----------
#     Y : numpy-array
#         Input matrix that is to be factorized. Y has shape (L,M), where L<=M.
#
#     sigma2 : int or None (default=None)
#         Variance of the noise on Y.
#
#     H : int or None (default = None)
#         Maximum rank of the factorized matrices.
#
#     Returns
#     -------
#     U : numpy-array
#         Left-singular vectors.
#
#     S : numpy-array
#         Diagonal matrix of singular values.
#
#     V : numpy-array
#         Right-singular vectors.
#
#     post : dictionary
#         Dictionary containing the computed posterior values.
#
#
#     References
#     ----------
#     .. [1] Nakajima, Shinichi, et al. "Global analytic solution of fully-observed variational Bayesian matrix factorization." Journal of Machine Learning Research 14.Jan (2013): 1-37.
#
#     .. [2] Nakajima, Shinichi, et al. "Perfect dimensionality recovery by variational Bayesian PCA." Advances in Neural Information Processing Systems. 2012.
#     """
#     L, M = Y.shape  # has to be L<=M
#
#     if H is None:
#         H = L
#
#     alpha = L / M
#     tauubar = 2.5129 * np.sqrt(alpha)
#
#     # SVD of the input matrix, max rank of H
#     U, s, V = np.linalg.svd(Y)
#     U = U[:, :H]
#     s = s[:H]
#     V = V[:H].T
#
#     # Calculate residual
#     residual = 0.
#     if H < L:
#         residual = np.sum(np.sum(Y ** 2) - np.sum(s ** 2))
#
#     # Estimation of the variance when sigma2 is unspecified
#     if sigma2 is None:
#         xubar = (1 + tauubar) * (1 + alpha / tauubar)
#         eH_ub = int(np.min([np.ceil(L / (1 + alpha)) - 1, H])) - 1
#         upper_bound = (np.sum(s ** 2) + residual) / (L * M)
#         lower_bound = np.max([s[eH_ub + 1] ** 2 / (M * xubar), np.mean(s[eH_ub + 1:] ** 2) / M])
#
#         scale = 1.  # /lower_bound
#         s = s * np.sqrt(scale)
#         residual = residual * scale
#         lower_bound = lower_bound * scale
#         upper_bound = upper_bound * scale
#
#         sigma2_opt = minimize_scalar(EVBsigma2, args=(L, M, s, residual, xubar), bounds=[lower_bound, upper_bound],
#                                      method='Bounded')
#         sigma2 = sigma2_opt.x
#
#     # Threshold gamma term
#     threshold = np.sqrt(M * sigma2 * (1 + tauubar) * (1 + alpha / tauubar))
#     pos = np.sum(s > threshold)
#
#     # Formula (15) from [2]
#     d = np.multiply(s[:pos] / 2, 1 - np.divide((L + M) * sigma2, s[:pos] ** 2) + np.sqrt(
#         (1 - np.divide((L + M) * sigma2, s[:pos] ** 2)) ** 2 - 4 * L * M * sigma2 ** 2 / s[:pos] ** 4))
#
#     # Computation of the posterior
#     post = {}
#     post['ma'] = np.zeros(H)
#     post['mb'] = np.zeros(H)
#     post['sa2'] = np.zeros(H)
#     post['sb2'] = np.zeros(H)
#     post['cacb'] = np.zeros(H)
#
#     tau = np.multiply(d, s[:pos]) / (M * sigma2)
#     delta = np.multiply(np.sqrt(np.divide(M * d, L * s[:pos])), 1 + alpha / tau)
#
#     post['ma'][:pos] = np.sqrt(np.multiply(d, delta))
#     post['mb'][:pos] = np.sqrt(np.divide(d, delta))
#     post['sa2'][:pos] = np.divide(sigma2 * delta, s[:pos])
#     post['sb2'][:pos] = np.divide(sigma2, np.multiply(delta, s[:pos]))
#     post['cacb'][:pos] = np.sqrt(np.multiply(d, s[:pos]) / (L * M))
#     post['sigma2'] = sigma2
#     post['F'] = 0.5 * (L * M * np.log(2 * np.pi * sigma2) + (residual + np.sum(s ** 2)) / sigma2
#                        + np.sum(M * np.log(tau + 1) + L * np.log(tau / alpha + 1) - M * tau))
#
#     return np.diag(d)
#
#
# def EVBsigma2(sigma2, L, M, s, residual, xubar):
#     H = len(s)
#
#     alpha = L / M
#     x = s ** 2 / (M * sigma2)
#
#     z1 = x[x > xubar]
#     z2 = x[x <= xubar]
#     tau_z1 = tau(z1, alpha)
#
#     term1 = np.sum(z2 - np.log(z2))
#     term2 = np.sum(z1 - tau_z1)
#     term3 = np.sum(np.log(np.divide(tau_z1 + 1, z1)))
#     term4 = alpha * np.sum(np.log(tau_z1 / alpha + 1))
#
#     obj = term1 + term2 + term3 + term4 + residual / (M * sigma2) + (L - H) * np.log(sigma2)
#
#     return obj
#
#
# def phi0(x):
#     return x - np.log(x)
#
#
# def phi1(x, alpha):
#     return np.log(tau(x, alpha) + 1) + alpha * np.log(tau(x, alpha) / alpha + 1) - tau(x, alpha)
#
#
# def tau(x, alpha):
#     return 0.5 * (x - (1 + alpha) + np.sqrt((x - (1 + alpha)) ** 2 - 4 * alpha))
