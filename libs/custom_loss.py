import torch
import torch.nn as nn
import numpy as np


class CustomLoss(nn.Module):

    def __init__(self, num_columns, reduction='mean', eps=1e-12):
        super(CustomLoss, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.reduction = reduction
        self.eps = eps
        self.num_columns = num_columns

    def forward(self, inputs, targets):
        """
        Compute loss.
        inputs (torch.Tensor): predicted labels (before sigmoid)
        targets (torch.Tensor): whole image true label
        """
        p = torch.sigmoid(inputs)
        q = self.make_q(p).to(self.device)

        # To avoid overflow:
        logp = torch.log(torch.clamp(p, min=self.eps))
        log_false = torch.log(torch.clamp(1 - p, min=self.eps))

        h = -targets * torch.sum(torch.mul(torch.matmul(p.unsqueeze(1), q).squeeze(1), logp), dim=1)
        l1 = -targets * torch.sum(log_false, dim=1) / self.num_columns
        # l1 = -targets * torch.max(log_false, dim=1).values
        l0 = -targets * torch.sum(logp, dim=1) / self.num_columns

        loss = h + l1 + l0
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss

    @staticmethod
    def make_q(values):
        q = []
        length = values.shape[-1]
        for ib in range(values.shape[0]):
            qb = []
            for i, val in enumerate(values[ib]):
                qb.append([val.item() if j == i + 1 or j == i - 1 else 0 for j in range(length)])
            q.append(np.array(qb).T)
        return torch.tensor(q).float()


class MultiHeadLoss(nn.Module):

    def __init__(self, num_columns, reduction='mean', eps=1e-12):
        super(MultiHeadLoss, self).__init__()
        self.reduction = reduction
        self.eps = eps
        self.num_columns = num_columns

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        # To avoid overflow:
        logp = torch.log(torch.clamp(p, min=self.eps))
        log_false = torch.log(torch.clamp(1 - p, min=self.eps))

        l0 = -targets * torch.max(logp, dim=1).values
        l1 = -(1 - targets) * torch.sum(log_false, dim=1) / self.num_columns

        loss = l0 + l1

        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss


class MultiHeadFlippedLossMSE(nn.Module):

    def __init__(self, num_columns, reduction='mean', eps=1e-12, lambda_mse=1):
        super(MultiHeadFlippedLossMSE, self).__init__()
        self.reduction = reduction
        self.eps = eps
        self.num_columns = num_columns
        self.lambda_mse = lambda_mse
        self.flipped_loss_function = nn.MSELoss(reduction='none')

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, flipped_inputs, targets):
        p = torch.sigmoid(inputs)
        q = torch.sigmoid(flipped_inputs)

        # To avoid overflow:
        logp = torch.log(torch.clamp(p, min=self.eps))
        log_false = torch.log(torch.clamp(1 - p, min=self.eps))

        l0 = -targets * torch.max(logp, dim=1).values
        l1 = -(1 - targets) * torch.sum(log_false, dim=1) / self.num_columns
        lflip = torch.sum(self.flipped_loss_function(inputs, torch.fliplr(q)), dim=1) / self.num_columns

        loss = l0 + l1 + self.lambda_mse * lflip

        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss


class MultiHeadFlippedLossKL(nn.Module):

    def __init__(self, num_columns, reduction='mean', eps=1e-12, lambda_kl=1):
        super(MultiHeadFlippedLossKL, self).__init__()
        self.reduction = reduction
        self.eps = eps
        self.num_columns = num_columns
        self.lambda_kl = lambda_kl
        self.kl_loss = nn.KLDivLoss(reduction='sum')

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, flipped_inputs, targets):
        p = torch.sigmoid(inputs)
        q = torch.sigmoid(flipped_inputs)
        qflip = torch.fliplr(q)

        # To avoid overflow:
        logp = torch.log(torch.clamp(p, min=self.eps))
        logq = torch.log(torch.clamp(q, min=self.eps))
        log_false_p = torch.log(torch.clamp(1 - p, min=self.eps))
        log_false_q = torch.log(torch.clamp(1 - q, min=self.eps))

        l22 = -targets * torch.max(logp, dim=1).values
        l21 = -(1 - targets) * torch.sum(log_false_p, dim=1) / self.num_columns

        l3 = 0.5 * (self.kl_loss(logp, qflip) + self.kl_loss(qflip.log(), p))

        l22_flip = -targets * torch.max(logq, dim=1).values
        l21_flip = -(1 - targets) * torch.sum(log_false_q, dim=1) / self.num_columns

        loss = l21 + l22 + self.lambda_kl * l3 + l22_flip + l21_flip

        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss
