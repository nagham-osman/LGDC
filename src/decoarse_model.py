import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
import wandb
import time
import os

from src.expansion import Expansion
from src.models.ppgn import PPGN
from src.diffusion.distributions import DistributionNodes
from models.graphformer_uncon import GraphTransformerUncon
from src.diffusion_model_discrete import DiscreteDenoisingDiffusion
import utils
from src.diffusion import diffusion_utils

from metrics.abstract_metrics import CrossEntropyMetric

class Decoarse(DiscreteDenoisingDiffusion):
    def __init__(self,
                 cfg, dataset_infos, train_metrics, sampling_metrics,
                 visualization_tools, extra_features, domain_features):
        super().__init__(cfg, dataset_infos, train_metrics, sampling_metrics,
                         visualization_tools, extra_features, domain_features)
        self.cfg = cfg
        self.dataset_info = dataset_infos
        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.node_dist = DistributionNodes(dataset_infos.original_n_nodes)

        if cfg.model.refine_type == 'oneshot':
            input_dimensions = self.dataset_info.output_dims
            output_dimensions = self.dataset_info.output_dims
        elif cfg.model.refine_type == 'diffusion':
            input_dimensions = self.dataset_info.input_dims
            output_dimensions = self.dataset_info.output_dims

        if cfg.model.refine_model == 'graphformer':
            self.decoarsen_model = GraphTransformerUncon(n_layers=self.cfg.model.n_layers,
                                                         input_dims=input_dimensions,
                                                         hidden_mlp_dims=self.cfg.model.hidden_mlp_dims,
                                                         hidden_dims=self.cfg.model.hidden_dims,
                                                         output_dims=output_dimensions,
                                                         act_fn_in=nn.ReLU(),
                                                         act_fn_out=nn.ReLU())
        elif cfg.model.refine_model == 'PPGN':
            self.decoarsen_model = PPGN(in_features=self.dataset_info.input_dims['E'],
                                        out_features=self.dataset_info.output_dims['E'],
                                        emb_features=self.cfg.decoarse.emb_features,
                                        hidden_features=self.cfg.decoarse.hidden_features,
                                        ppgn_features=self.cfg.decoarse.ppgn_features,
                                        num_layers=self.cfg.decoarse.num_layers,
                                        dropout=self.cfg.decoarse.dropout)

        self.decoarsen_edge_loss = TrainLossDiscreteEdge(self.cfg.model.lambda_train)

        self.expansion = Expansion(
            node_features=self.dataset_info.output_dims['X'],
            augmented_radius=cfg.reduction.augmented_radius,
            augmented_dropout=cfg.reduction.augmented_dropout,
            deterministic_expansion=cfg.reduction.deterministic_expansion,
            min_red_frac=cfg.reduction.min_red_frac,
            max_red_frac=cfg.reduction.max_red_frac,
            red_threshold=cfg.reduction.red_threshold,
        )
        self.expansion.to('cuda:' + str(cfg.general.gpu_id[0]))

        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics'])
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0

    def forward(self, data, node_mask):
        X = data.X
        E = data.E
        if self.cfg.model.refine_model == 'graphformer':
            if self.cfg.model.refine_type == 'oneshot':
                if self.cfg.model.uncon_gen is False:
                    pred = self.decoarsen_model(X, E, y, node_mask)
                else:
                    pred = self.decoarsen_model(X, E, node_mask)
            elif self.cfg.model.refine_type == 'diffusion':
                noisy_data = self.apply_noise(X, E, None, node_mask)
                extra_data = self.compute_extra_data(noisy_data)
                X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
                E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
                if self.cfg.model.uncon_gen is False:
                    y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
                    pred = self.decoarsen_model(X, E, y, node_mask_aug)
                else:
                    pred = self.decoarsen_model(X, E, node_mask_aug)
        elif self.cfg.model.refine_model == 'PPGN':
            e_mask_aug = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)
            e_mask_aug = e_mask_aug.unsqueeze(-1).float()
            if self.cfg.model.refine_type == 'oneshot':
                pred_edge = self.decoarsen_model(E, e_mask_aug,
                                                 noise_cond=torch.zeros(E.shape[0]).to(
                                                    E.device))
            elif self.cfg.model.refine_type == 'diffusion':
                noisy_data = self.apply_noise(X, E, None, node_mask)
                extra_data = self.compute_extra_data(noisy_data)
                X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
                E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
                e_mask_aug = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)
                e_mask_aug = e_mask_aug.unsqueeze(-1).float()
                pred_edge = self.decoarsen_model(E, e_mask_aug,
                                                 noise_cond=torch.zeros(E.shape[0]).to(
                                                     E.device))
            pred = utils.PlaceHolder(X=X, E=pred_edge, y=None)
        return pred


    def training_step(self, data, i):
        """
            1) Use the coarsened version of the original graph
            2) Decoarsen (expand) it back
            3) Return the reconstructed data
        """
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return

        # ground truth
        dense_data_true, _ = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)

        # coarsened data construction
        node_feature = F.one_hot(
            data.reduced_node_expansion.to(torch.long) - 1,
            num_classes=self.dataset_info.max_expansion_size
        )

        dense_data, node_mask = utils.to_dense_coarsen(
            node_feature,
            data.adj_reduced,
            data.batch_reduced
        )
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        node_emb, E_augmented, batch = self.expansion.expand(
            E,
            data.batch_reduced,
            data.reduced_node_expansion,
            data.target_size
        )
        X_augmented, node_mask_aug = utils.to_dense_batch(node_emb, data.batch)
        dense_data_aug = utils.PlaceHolder(X=X_augmented, E=E_augmented, y=None)
        pred = self.forward(dense_data_aug, node_mask_aug)

        loss = self.decoarsen_edge_loss(masked_pred_E=pred.E,
                                        true_E=dense_data_true.E,
                                        log=i % self.log_every_steps == 0)

        self.train_metrics(masked_pred_X=pred.X, masked_pred_E=pred.E, true_X=dense_data_true.X,
                           true_E=dense_data_true.E,
                           log=i % self.log_every_steps == 0)

        self.log("train_loss", loss, on_epoch=True, on_step=True)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        print(f"unconditional generation: {self.cfg.model.uncon_gen}")

        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.decoarsen_edge_loss.reset()
        self.train_metrics.reset()

    def validation_step(self, data, i):
        dense_data_true, _ = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        reduced_node_expansion = F.one_hot(
            data.reduced_node_expansion.to(torch.long) - 1,
            num_classes=self.dataset_info.max_expansion_size
        )
        dense_data, node_mask = utils.to_dense_coarsen(
            reduced_node_expansion,
            data.adj_reduced,
            data.batch_reduced
        )
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        node_emb, E_augmented, batch = self.expansion.expand(
            E,
            data.batch_reduced,
            data.reduced_node_expansion,
            data.target_size
        )
        X_augmented, node_mask_aug = utils.to_dense_batch(node_emb, data.batch)
        dense_data_aug = utils.PlaceHolder(X=X_augmented, E=E_augmented, y=None)
        pred = self.forward(dense_data_aug, node_mask_aug)

        loss = self.decoarsen_edge_loss(masked_pred_E=pred.E,
                                        true_E=dense_data_true.E,
                                        log=i % self.log_every_steps == 0)

        pred = pred.mask(node_mask_aug, collapse=True)
        if not hasattr(self, "validation_step_outputs"):
            self.validation_step_outputs = []
        self.validation_step_outputs.append({'loss': loss, 'pred': pred})

        return {'loss': loss,
                'pred': pred}

    def on_validation_epoch_end(self) -> None:
        if not hasattr(self, "validation_step_outputs"):
            self.validation_step_outputs = []  # Initialize if not set
        # Aggregate losses
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        if wandb.run:
            wandb.log({"avg_val_loss": avg_loss}, commit=False)

        molecule_list = []
        for out in self.validation_step_outputs:
            pred = out["pred"]

            # If your batch size is 1 or you return a single-graph pred:
            if pred.X.dim() == 2:
                atom_types = pred.X.cpu()
                edge_types = pred.E.cpu()
                molecule_list.append([atom_types, edge_types])
            else:
                bsz = pred.X.size(0)
                for b in range(bsz):
                    atom_types_b = pred.X[b].cpu()  # shape [N, x_dim]
                    edge_types_b = pred.E[b].cpu()  # shape [N, N]
                    print(atom_types_b.shape)
                    print(edge_types_b.shape)

                    molecule_list.append([atom_types_b, edge_types_b])

        self.validation_step_outputs.clear()

        self.val_counter += 1
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            self.print("Computing sampling metrics...")
            self.sampling_metrics.forward(
                molecule_list,
                self.name,
                self.current_epoch,
                val_counter=-1,
                test=False,
                local_rank=self.local_rank
            )
            self.print(f'Done. Sampling took {time.time() - start:.2f} seconds\n')

            if self.visualization_tools is not None:
                self.print("Visualizing molecules...")

                current_path = os.getcwd()
                result_path = os.path.join(current_path, f'graphs/{self.name}/epoch{self.current_epoch}_validation/')
                if not os.path.exists(result_path):
                    os.makedirs(result_path)

                # Visualize molecules and save results
                self.visualization_tools.visualize(result_path, molecule_list, 4)

                self.print(f"Visualization complete. Graphs saved to: {result_path}")

    def test_step(self, data, batch_idx):
        dense_data_true, _ = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        reduced_node_expansion = F.one_hot(
            data.reduced_node_expansion.to(torch.long) - 1,
            num_classes=self.dataset_info.max_expansion_size
        )
        dense_data, node_mask = utils.to_dense_coarsen(
            reduced_node_expansion,
            data.adj_reduced,
            data.batch_reduced
        )
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        node_emb, E_augmented, batch = self.expansion.expand(
            E,
            data.batch_reduced,
            data.reduced_node_expansion,
            data.target_size
        )
        X_augmented, node_mask_aug = utils.to_dense_batch(node_emb, data.batch)
        dense_data_aug = utils.PlaceHolder(X=X_augmented, E=E_augmented, y=None)
        pred = self.forward(dense_data_aug, node_mask_aug)

        loss = self.decoarsen_edge_loss(masked_pred_E=pred.E,
                                        true_E=dense_data_true.E,
                                        log=i % self.log_every_steps == 0)

        pred = pred.mask(node_mask_aug, collapse=True)
        if not hasattr(self, "test_step_outputs"):
            self.test_step_outputs = []
        self.test_step_outputs.append({'loss': loss, 'pred': pred})

        return {'loss': loss,
                'pred': pred}

    def on_test_epoch_end(self) -> None:
        if not hasattr(self, "test_step_outputs"):
            self.test_step_outputs = []  # Initialize if not set
        # Aggregate losses
        avg_loss = torch.stack([x['loss'] for x in self.test_step_outputs]).mean()
        if wandb.run:
            wandb.log({"avg_test_loss": avg_loss}, commit=False)

        molecule_list = []
        for out in self.test_step_outputs:
            pred = out["pred"]

            # If your batch size is 1 or you return a single-graph pred:
            if pred.X.dim() == 2:
                atom_types = pred.X.cpu()
                edge_types = pred.E.cpu()
                molecule_list.append([atom_types, edge_types])
            else:
                bsz = pred.X.size(0)
                for b in range(bsz):
                    atom_types_b = pred.X[b].cpu()  # shape [N, x_dim]
                    edge_types_b = pred.E[b].cpu()  # shape [N, N]

                    molecule_list.append([atom_types_b, edge_types_b])

        self.test_step_outputs.clear()

        self.print("Computing sampling metrics...")
        self.sampling_metrics.forward(
            molecule_list,
            self.name,
            self.current_epoch,
            val_counter=-1,
            test=False,
            local_rank=self.local_rank
        )
        self.print(f'Done. Sampling took {time.time() - start:.2f} seconds\n')

        if self.visualization_tools is not None:
            self.print("Visualizing molecules...")

            current_path = os.getcwd()
            result_path = os.path.join(current_path, f'graphs/{self.name}/epoch{self.current_epoch}_validation/')
            if not os.path.exists(result_path):
                os.makedirs(result_path)

            # Visualize molecules and save results
            self.visualization_tools.visualize(result_path, molecule_list, 4)

            self.print(f"Visualization complete. Graphs saved to: {result_path}")

    def one_shot_refine(self, X, E, y, node_mask):
        if self.cfg.model.refine_model == 'graphformer':
            if self.cfg.model.uncon_gen is False:
                pred = self.decoarsen_model(X, E, y, node_mask)
            else:
                pred = self.decoarsen_model(X, E, node_mask)
        elif self.cfg.model.refine_model == 'PPGN':
            e_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
            e_mask = e_mask.unsqueeze(-1).float()
            pred_E = self.decoarsen_model(E, e_mask,
                                          noise_cond=torch.zeros(E.shape[0]).to(E.device))
            pred = utils.PlaceHolder(X=X, E=pred_E, y=None)
        pred_X = F.softmax(pred.X, dim=-1)
        pred_E = F.softmax(pred.E, dim=-1)
        out_one_hot = utils.PlaceHolder(X=pred_X, E=pred_E, y=torch.zeros(y.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=pred_X, E=pred_E, y=torch.zeros(y.shape[0], 0))
        return out_one_hot.mask(node_mask).type_as(y), out_discrete.mask(node_mask, collapse=True).type_as(y)


class TrainLossDiscreteEdge(nn.Module):
    """ Train with Cross entropy"""
    def __init__(self, lambda_train):
        super().__init__()
        self.edge_loss = CrossEntropyMetric()
        self.lambda_train = lambda_train

    def forward(self, masked_pred_E, true_E, log: bool):
        """ Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        log : boolean. """
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)

        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0

        if log:
            to_log = {"train_loss/De_E_CE": self.edge_loss.compute() if true_E.numel() > 0 else -1}
            if wandb.run:
                wandb.log(to_log, commit=True)
        return self.lambda_train[0] * loss_E

    def reset(self):
        self.edge_loss.reset()