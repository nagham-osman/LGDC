import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric, MeanSquaredError, MetricCollection
import time
import wandb
from src.metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchMSE, SumExceptBatchKL, CrossEntropyMetric, \
    ProbabilityMetric, NLL


class NodeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class EdgeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class TrainLoss(nn.Module):
    def __init__(self, use_y):
        super(TrainLoss, self).__init__()
        self.train_node_mse = NodeMSE()
        self.train_edge_mse = EdgeMSE()
        self.train_y_mse = MeanSquaredError()
        self.use_y = use_y

    def forward(self, masked_pred_epsX, masked_pred_epsE, pred_y, true_epsX, true_epsE, true_y, log: bool):
        mse_X = self.train_node_mse(masked_pred_epsX, true_epsX) if true_epsX.numel() > 0 else 0.0
        mse_E = self.train_edge_mse(masked_pred_epsE, true_epsE) if true_epsE.numel() > 0 else 0.0
        if self.use_y:
            mse_y = self.train_y_mse(pred_y, true_y) if true_y.numel() > 0 else 0.0
        else:
            mse_y = 0
        mse = mse_X + mse_E + mse_y

        if log:
            to_log = {'train_loss/batch_mse': mse.detach(),
                      'train_loss/node_MSE': self.train_node_mse.compute(),
                      'train_loss/edge_MSE': self.train_edge_mse.compute(),
                      'train_loss/y_mse': self.train_y_mse.compute() if self.use_y and (true_y.numel() > 0) else -1}
            if wandb.run:
                wandb.log(to_log, commit=True)

        return mse

    def reset(self):
        for metric in (self.train_node_mse, self.train_edge_mse, self.train_y_mse):
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_mse = self.train_node_mse.compute() if self.train_node_mse.total > 0 else -1
        epoch_edge_mse = self.train_edge_mse.compute() if self.train_edge_mse.total > 0 else -1
        epoch_y_mse = self.train_y_mse.compute() if self.train_y_mse.total > 0 else -1

        to_log = {"train_epoch/epoch_X_mse": epoch_node_mse,
                  "train_epoch/epoch_E_mse": epoch_edge_mse,
                  "train_epoch/epoch_y_mse": epoch_y_mse}
        if wandb.run:
            wandb.log(to_log)
        return to_log



class TrainLossDiscrete(nn.Module):
    """ Train with Cross entropy"""
    def __init__(self, lambda_train, use_y):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        self.lambda_train = lambda_train
        self.use_y = use_y

    def forward(self, masked_pred_X, masked_pred_E, pred_y, true_X, true_E, true_y, log: bool):
        """ Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        log : boolean. """
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)

        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        loss_X = self.node_loss(flat_pred_X, flat_true_X) if true_X.numel() > 0 else 0.0
        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0
        if self.use_y:
            loss_y = self.y_loss(pred_y, true_y) if true_y.numel() > 0 else 0.0
        else:
            loss_y = 0

        if log:
            to_log = {"train_loss/batch_CE": (loss_X + loss_E + loss_y).detach(),
                      "train_loss/X_CE": self.node_loss.compute() if true_X.numel() > 0 else -1,
                      "train_loss/E_CE": self.edge_loss.compute() if true_E.numel() > 0 else -1,
                      "train_loss/y_CE": self.y_loss.compute() if self.use_y and true_y.numel() > 0 else -1}
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss_X + self.lambda_train[0] * loss_E + self.lambda_train[1] * loss_y

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        epoch_edge_loss = self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        epoch_y_loss = self.train_y_loss.compute() if self.y_loss.total_samples > 0 else -1

        to_log = {"train_epoch/x_CE": epoch_node_loss,
                  "train_epoch/E_CE": epoch_edge_loss,
                  "train_epoch/y_CE": epoch_y_loss}
        if wandb.run:
            wandb.log(to_log, commit=False)

        return to_log


# class TrainLossESGG:
#     def __init__(self):
#         super().__init__()
#
#     def get_loss(self, x, mask, x_pred, log: bool):
#         """Compute loss to train the model.
#
#         Sample x_pred ~ p(x_t, t), where t ~ U(0, T-1) and x_t ~ q(x_t | x),
#         and compute the cross entropy loss between x_pred and x.
#         """
#         # x assumed to contain discrete labels
#         x = x.long()  # N, n, n
#
#         # masks
#         n = mask.shape[1]
#         mask = mask.long()
#         mask_diag = 1 - th.eye(n, device=mask.device, dtype=th.long).view(1, n, n)
#
#         # compute loss
#         loss_mask = (mask * mask_diag).float()
#         loss = self.loss_compute(x, x_pred) * loss_mask
#         loss = loss.sum() / loss_mask.sum()
#
#         if log:
#             to_log = {"train_loss/De_E_CE": self.edge_loss.compute() if true_E.numel() > 0 else -1}
#             if wandb.run:
#                 wandb.log(to_log, commit=True)
#         return loss
#
#     def loss_compute(self, x, pred):
#         return F.cross_entropy(
#             pred.view(-1, pred.shape[-1]), x.view(-1), reduction="none"
#         ).view(x.shape)
#
#     def reset(self):
#         self.edge_loss.reset()


class EdgeLossMetric:
    def __init__(self):
        self.total_loss = 0.0
        self.count = 0

    def update(self, loss):
        self.total_loss += loss.item()
        self.count += 1

    def compute(self):
        return self.total_loss / self.count if self.count > 0 else 0.0

    def reset(self):
        self.total_loss = 0.0
        self.count = 0


class TrainLossESGG(nn.Module):
    def __init__(self, weights_train):
        super().__init__()
        self.edge_loss_metric = EdgeLossMetric()
        # self.weights = torch.tensor(weights_train)
        self.weights = torch.tensor(weights_train, dtype=torch.float32)

    def forward(self, masked_pred_E, true_E, node_mask, log: bool):
        """
        Compute loss to train the model.

        x: ground truth edge labels. No edge is a special type.
        mask: a node mask or edge mask. If it is a node mask (B, n), convert it to an edge mask (B, n, n).
        x_pred: predicted logits with shape (B, n, n, num_classes).
        """
        if true_E.ndim == 4 and true_E.shape[-1] > 1:
            true_E = torch.argmax(true_E, dim=-1)
        true_E = true_E.long()

        if node_mask.ndim == 2:
            node_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        node_mask = node_mask.long()

        n = node_mask.shape[1]
        mask_diag = 1 - torch.eye(n, device=node_mask.device, dtype=torch.long).view(1, n, n)
        loss_mask = (node_mask * mask_diag).float()

        weights = self.weights.to(masked_pred_E.device)
        # loss_tensor = self.loss_compute(true_E, masked_pred_E) * loss_mask
        loss_tensor = self.loss_compute(true_E, masked_pred_E, weights) * loss_mask
        loss = loss_tensor.sum() / loss_mask.sum()

        self.edge_loss_metric.update(loss)

        if log:
            log_value = self.edge_loss_metric.compute() if true_E.numel() > 0 else -1
            to_log = {"train_loss/De_E_CE": log_value}
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss

    def loss_compute(self, x, pred, weight):
        """
        Compute the per-element cross entropy loss.
        x: (B, n, n) with discrete labels.
        pred: (B, n, n, num_classes)
        """
        # return F.cross_entropy(
        #     pred.view(-1, pred.shape[-1]), x.view(-1), reduction="none"
        # ).view(x.shape)

        return F.cross_entropy(
            pred.view(-1, pred.shape[-1]),
            x.view(-1),
            reduction="none",
            weight=weight
        ).view(x.shape)

    def reset(self):
        self.edge_loss_metric.reset()




class TrainLossDiscreteEdge(nn.Module):
    """ Train with Cross entropy"""
    def __init__(self, lambda_train):
        super().__init__()
        self.edge_loss = CrossEntropyMetric()
        self.lambda_train = lambda_train

    def forward(self, masked_pred_E, true_E, log: bool):
        """ Compute train metrics
        masked_pred_E : tensor -- (bs, n, n, de)
        true_E : tensor -- (bs, n, n, de)
        log : boolean. """
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)

        mask_E = (true_E != 0.).any(dim=-1)
        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        mask_E = (flat_true_E[:, 1] == 1)
        flat_true_E = flat_true_E[mask_E, :]
        flat_pred_E = flat_pred_E[mask_E, :]

        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0

        if log:
            to_log = {"train_loss/De_E_CE": self.edge_loss.compute() if true_E.numel() > 0 else -1}
            if wandb.run:
                wandb.log(to_log, commit=True)
        return self.lambda_train[0] * loss_E

    def reset(self):
        self.edge_loss.reset()




class TrainLossDiscreteEdgeSparse(nn.Module):
    """ Train with Cross entropy on sparse edge predictions """
    def __init__(self, lambda_train):
        super().__init__()
        self.edge_loss = CrossEntropyMetric()
        self.lambda_train = lambda_train

    def forward(self, pred_E: torch.Tensor, true_E: torch.Tensor, log: bool):
        """
        pred_E: (E, de) - predicted logits or probabilities for each of E edges
        true_E: (E, de) or (E,) - ground-truth labels for each edge
        log: bool - whether to log metrics

        For a multi-class edge setting:
        - If 'true_E' is one-hot => shape (E, de)
        - If 'true_E' is integer class => shape (E,)
        """
        if true_E.numel() == 0 or pred_E.numel() == 0:
            loss_E = 0.0
        else:
            loss_E = self.edge_loss(pred_E, true_E)

        if log:
            to_log = {"train_loss/Sparse_E_CE": self.edge_loss.compute() if true_E.numel() > 0 else -1}
            import wandb
            if wandb.run:
                wandb.log(to_log, commit=True)

        return self.lambda_train[0] * loss_E

    def reset(self):
        self.edge_loss.reset()
