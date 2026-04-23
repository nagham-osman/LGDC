import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
import os
import socket
import gc
import torch.distributed as dist
from torch_geometric.utils import to_edge_index
from torch_scatter import scatter
from torch_sparse import SparseTensor

from diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete,\
    MarginalUniformTransition
from src.diffusion.distributions import DistributionNodes
from src.diffusion import diffusion_utils
from src.diffusion_model_discrete import DiscreteDenoisingDiffusion
from src.expansion import Expansion
from src.models.sparse_ppgn import SparsePPGN
from src.models.ppgn import PPGN
from src.models.sign_net import SignNet
from src.graph_coarsen.spectral import SpectrumExtractor
from src.expansion.decoarsen_diffusion import DiscreteGraphDiffusion
from models.graphformer_uncon import GraphTransformerUncon

from metrics.train_metrics import TrainLossDiscrete, TrainLossDiscreteEdge, TrainLossDiscreteEdgeSparse, TrainLossESGG
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL, CrossEntropyMetric
from src import utils


class CoarsenedDDM(DiscreteDenoisingDiffusion):
    def __init__(self,
                 cfg, dataset_infos, train_metrics, sampling_metrics,
                 visualization_tools, extra_features, domain_features):
        super().__init__(cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features)

        self.start_epoch_time = time.time()

        self.dataset_info = dataset_infos
        if cfg.model.refine_type == 'oneshot':
            input_dimensions = self.dataset_info.output_dims
            output_dimensions = self.dataset_info.output_dims
        elif cfg.model.refine_type == 'diffusion':
            input_dimensions = self.dataset_info.input_dims
            output_dimensions = self.dataset_info.output_dims

        output_dimensions_decoarse = self.dataset_info.output_dims_decoarse

        self.Xdim = input_dimensions['X']
        self.Xdim_output = output_dimensions['X']
        self.Edim = input_dimensions['E']
        self.Edim_output = output_dimensions['E']
        self.Xdim_decoarse = output_dimensions_decoarse['X']
        self.Edim_decoarse = output_dimensions_decoarse['E']

        if self.cfg.model.refine_model == 'graphformer':
            self.decoarsen_model = GraphTransformerUncon(n_layers=cfg.model.n_layers,
                                                         input_dims=input_dimensions,
                                                         hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                                         hidden_dims=cfg.model.hidden_dims,
                                                         output_dims=output_dimensions,
                                                         act_fn_in=nn.ReLU(),
                                                         act_fn_out=nn.ReLU())
        elif self.cfg.model.refine_model == 'sparsePPGN':
            self.decoarsen_model = SparsePPGN(node_in_features=input_dimensions['X'] * (1 + cfg.decoarse.self_conditioning),
                                              edge_in_features=input_dimensions['E'] * (1 + cfg.decoarse.self_conditioning),
                                              node_out_features=output_dimensions['X'],
                                              edge_out_features=output_dimensions['E'],
                                              emb_features=self.cfg.decoarse.emb_features,
                                              hidden_features=self.cfg.decoarse.hidden_features,
                                              ppgn_features=self.cfg.decoarse.ppgn_features,
                                              num_layers=self.cfg.decoarse.num_layers,
                                              dropout=self.cfg.decoarse.dropout)
        elif self.cfg.model.refine_model == 'PPGN':
            self.decoarsen_model = PPGN(in_features=self.dataset_info.input_dims['E'],
                                        out_features=self.dataset_info.output_dims['E'],
                                        emb_features=self.cfg.decoarse.emb_features,
                                        hidden_features=self.cfg.decoarse.hidden_features,
                                        ppgn_features=self.cfg.decoarse.ppgn_features,
                                        num_layers=self.cfg.decoarse.num_layers,
                                        dropout=self.cfg.decoarse.dropout)

        if not cfg.train.pretrain_decoarsen and cfg.general.decoarse_checkpoint is not None:
            print('Loading Decoarsening Model from checkpoint...')
            checkpoint_path = cfg.general.decoarse_checkpoint
            if not dist.is_initialized():
                master_addr = "127.0.0.1"
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('', 0))
                port = sock.getsockname()[1]
                sock.close()
                master_port = port
                init_method = f"tcp://{master_addr}:{master_port}"
                dist.init_process_group(
                    backend="gloo",
                    init_method=init_method,
                    rank=0,
                    world_size=1
                )

            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            decoarsen_state_dict = {
                key[len('decoarsen_model.'):]: value
                for key, value in checkpoint['state_dict'].items()
                if key.startswith('decoarsen_model.')
            }
            self.decoarsen_model.load_state_dict(decoarsen_state_dict)

            self.decoarsen_model.eval()
            for param in self.decoarsen_model.parameters():
                param.requires_grad = False
            print('Successfully loaded!')

        self.decoarsen_diffusion = DiscreteGraphDiffusion(
            X_dim=self.Xdim,
            E_dim=self.Edim,
            self_conditioning=cfg.decoarse.self_conditioning,
            num_steps=cfg.decoarse.num_steps,
            sampling_mode = cfg.reduction.sampling_mode,
        )

        self.expansion = Expansion(
            augmented_radius=cfg.reduction.augmented_radius,
            augmented_dropout=cfg.reduction.augmented_dropout,
            deterministic_expansion=cfg.reduction.deterministic_expansion,
        )

        if cfg.spectral.num_features > 0:
            self.sign_net = SignNet(
                num_eigenvectors=cfg.spectral.num_features,
                hidden_features=cfg.sign_net.hidden_features,
                out_features=cfg.decoarse.emb_features,
                num_layers=cfg.sign_net.num_layers,
            )
        else:
            self.sign_net = None

        self.spectrum_extractor = (
            SpectrumExtractor(
                num_features=cfg.spectral.num_features,
                normalized=cfg.spectral.normalized_laplacian,
            )
            if cfg.spectral.num_features > 0 else None
        )

        ema_decay = cfg.train.ema_decay
        if ema_decay > 0:
            self.ema = utils.EMA(ema_decay)
            self.decoarsen_model_ema = copy.deepcopy(self.decoarsen_model)
            self.decoarsen_model_ema.to(self.device)
            for p in self.decoarsen_model_ema.parameters():
                p.requires_grad = False
        else:
            self.ema = None
            self.decoarsen_model_ema = None

        self.orig_node_dist = DistributionNodes(self.dataset_info.original_n_nodes)
        self.coarse_node_dist = DistributionNodes(self.dataset_info.n_nodes)

        self.val_decoarsen_nll = NLL()
        self.test_decoarsen_nll = NLL()
        self.validation_outputs = []
        self.test_outputs = []

    def on_load_checkpoint(self, checkpoint):
        """ Lightning calls this right after load_from_checkpoint() finishes. """
        super().on_load_checkpoint(checkpoint)

        ckpt_path = self.cfg.general.decoarse_checkpoint
        if ckpt_path:
            ckpt2 = torch.load(ckpt_path, map_location=self.device)
            sd2 = ckpt2["state_dict"]
            deco_sd = {}
            for k, v in sd2.items():
                if k.startswith("decoarsen_model."):
                    deco_sd[k[len("decoarsen_model."):]] = v
            missing, unexpected = self.decoarsen_model.load_state_dict(deco_sd, strict=False)
            print(f"Loaded decoarse ckpt `{ckpt_path}` → missing keys={missing}, unexpected={unexpected}")
            self.decoarsen_model.eval()
            for p in self.decoarsen_model.parameters():
                p.requires_grad = False

    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return

        if self.cfg.train.pretrain_decoarsen:
            loss_decoarse, loss_terms = self.get_loss_decoarsen(data.adj_reduced, data, self.decoarsen_model, self.sign_net)
            loss_c = 0.0
        else:
            # Full Pipeline Training (Diffusion + Decoarsening)
            # Training models separately
            # Compute diffusion loss on the coarsened graph
            pred_c, dense_data_c, node_mask = self.get_pred(data.reduced_node_expansion, data.adj_reduced,
                                                            data.batch_reduced)

            loss_c = self.train_loss(masked_pred_X=pred_c.X, masked_pred_E=pred_c.E, pred_y=pred_c.y,
                                     true_X=dense_data_c.X, true_E=dense_data_c.E, true_y=data.y,
                                     log=i % self.log_every_steps == 0)
            self.train_metrics(masked_pred_X=pred_c.X, masked_pred_E=pred_c.E, true_X=dense_data_c.X,
                               true_E=dense_data_c.E,
                               log=i % self.log_every_steps == 0)

            loss_decoarse, loss_terms = self.get_loss_decoarsen(data.adj_reduced, data, self.decoarsen_model,
                                                                self.sign_net)

        if i % self.log_every_steps == 0:
            to_log = {"train_loss/De_E_CE": loss_decoarse}
            if wandb.run:
                wandb.log(to_log, commit=True)

        total_loss = loss_c + loss_decoarse
        return {'loss': total_loss}


    def configure_optimizers(self):
        decoarse_trainable = any(p.requires_grad for p in self.decoarsen_model.parameters())
        if not decoarse_trainable:
            trainable_params = [p for p in self.parameters() if p.requires_grad]
            return torch.optim.Adam(trainable_params, lr=self.cfg.train.lr)
        else:
            lr1, wd1 = self.cfg.train.lr, self.cfg.train.weight_decay
            lr2, wd2 = self.cfg.decoarse.lr_dec, self.cfg.decoarse.w_decay

            return torch.optim.Adam([
                {
                    'params': self.model.parameters(),
                    'lr': lr1,
                    'weight_decay': wd1,
                },
                {
                    'params': self.decoarsen_model.parameters(),
                    'lr': lr2,
                    'weight_decay': wd2,
                },
            ])

    def get_loss_decoarsen(self, adj_reduced, batch, model, sign_net):
        """Returns a weighted sum of the node expansion loss and the augmented edge loss."""
        # get augmented graph
        adj_augmented = self.expansion.get_augmented_graph(
            adj_reduced, batch.expansion_matrix
        )

        # construct labels
        node_attr = batch.node_expansion - 1
        augmented_edge_index, edge_val = to_edge_index(adj_augmented + batch.adj)
        augmented_edge_attr = edge_val.long() - 1

        # get node embeddings
        if sign_net is not None:
            node_emb_reduced = sign_net(
                spectral_features=batch.spectral_features_reduced,
                edge_index=adj_reduced,
            )
            node_emb = batch.expansion_matrix @ node_emb_reduced
        else:
            node_emb = torch.randn(
                adj_augmented.size(0), self.cfg.decoarse.emb_features, device=self.device
            )

        # reduction fraction
        size = scatter(torch.ones_like(batch.batch), batch.batch)
        expanded_size = scatter(batch.node_expansion, batch.batch)
        red_frac = 1 - size / expanded_size

        # loss
        node_loss, edge_loss = self.decoarsen_diffusion.get_loss(
            edge_index=augmented_edge_index,
            batch=batch.batch,
            node_attr=node_attr,
            edge_attr=augmented_edge_attr,
            model=model,
            model_kwargs={
                "node_emb": node_emb,
                "red_frac": red_frac,
                "target_size": batch.target_size.float(),
            },
        )

        # ignore node_loss for first level
        # node_loss = node_loss[batch.reduction_level[batch.batch] > 0].mean()
        node_loss = node_loss.mean()
        edge_loss = edge_loss.mean()

        loss = node_loss + edge_loss

        return loss, {
            "node_expansion_loss": node_loss.item(),
            "augmented_edge_loss": edge_loss.item(),
            "loss": loss.item(),
        }


    def get_pred(self, node_feature, adj, batch):
        node_feature = F.one_hot(
            node_feature.to(torch.long) - 1,
            num_classes=self.dataset_info.max_expansion_size
        )

        dense_data, node_mask = utils.to_dense_coarsen(
            node_feature,
            adj,
            batch
        )
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, None, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        return pred, dense_data, node_mask


    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()


    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: X_CE: {to_log['train_epoch/x_CE'] :.3f}"
                      f" -- E_CE: {to_log['train_epoch/E_CE'] :.3f} --"
                      f" y_CE: {to_log['train_epoch/y_CE'] :.3f}"
                      f" -- {time.time() - self.start_epoch_time:.1f}s ")
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: {epoch_at_metrics} -- {epoch_bond_metrics}")
        print(torch.cuda.memory_summary())
        if self.ema is not None:
            self.ema.update_model_average(
                self.decoarsen_model_ema, self.decoarsen_model
            )

    def validation_step(self, data, i):
        assert not self.training, "Lightning should have set model.eval() for val"

        if self.cfg.train.pretrain_decoarsen:
            # Pretraining Decoarsening Module
            eval_model = self.decoarsen_model_ema if (self.ema and not self.training) else self.decoarsen_model
            loss_decoarse, loss_terms = self.get_loss_decoarsen(data.adj_reduced, data, eval_model, self.sign_net)
            loss_c = 0.0
            self.val_decoarsen_nll(loss_decoarse)
        else:
            # Full Pipeline Training (Diffusion + Decoarsening)
            # Compute diffusion loss on the coarsened graph
            pred_c, dense_data_c, node_mask = self.get_pred(data.reduced_node_expansion, data.adj_reduced,
                                                            data.batch_reduced)

            loss_c = self.train_loss(masked_pred_X=pred_c.X, masked_pred_E=pred_c.E, pred_y=pred_c.y,
                                     true_X=dense_data_c.X, true_E=dense_data_c.E, true_y=data.y,
                                     log=False)

            eval_model = self.decoarsen_model_ema if (self.ema and not self.training) else self.decoarsen_model
            loss_decoarse, loss_terms = self.get_loss_decoarsen(data.adj_reduced, data, eval_model, self.sign_net)
            self.val_nll(loss_c)
            self.val_decoarsen_nll(loss_decoarse)

        total_loss = loss_c + loss_decoarse

        if wandb.run:
            wandb.log({
                "val/diffusion_loss": loss_c,
                "val/decoarsening_loss": loss_decoarse,
                "val/total_loss": total_loss,
            }, commit=False)

        if self.cfg.train.pretrain_decoarsen:
            batch_size, _, _ = dense_data.X.shape
            output = {"dense_data": dense_data, "node_mask": node_mask,
                      "batch_size": batch_size, 'loss': total_loss}
            self.validation_outputs.append(output)

        return {'loss': total_loss, 'diffusion_loss': loss_c, 'decoarsening_loss': loss_decoarse}


    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.sampling_metrics.reset()
        self.val_decoarsen_nll.reset()


    def on_validation_epoch_end(self) -> None:
        if self.cfg.train.pretrain_decoarsen:
            decoarse_nll = self.val_decoarsen_nll.compute()
            if wandb.run:
                wandb.log({"val/epoch_decoarse_NLL": decoarse_nll}, commit=False)
            self.print(f"Epoch {self.current_epoch}: Val Decoarsen NLL: {decoarse_nll:.2f}")
            self.log("val/epoch_NLL", decoarse_nll, sync_dist=True)

            self.val_counter += 1
            if self.val_counter % self.cfg.general.sample_every_val == 0:
                start = time.time()
                pred_graphs = []
                all_target_n_nodes = []
                outputs = self.validation_outputs
                print('Start sampling...')
                for graph_data in outputs:
                    dense_data = graph_data["dense_data"]
                    node_mask = graph_data["node_mask"]
                    batch_size = graph_data["batch_size"]
                    target_n_nodes = self.orig_node_dist.sample_n(batch_size, self.device)
                    all_target_n_nodes.append(target_n_nodes)
                    eval_model = self.decoarsen_model_ema if (self.ema and not self.training) else self.decoarsen_model
                    pred_graphs_batch = self.sample_pred_graphs(dense_data.X, dense_data.E, node_mask,
                                                                target_n_nodes, self.decoarsen_diffusion,
                                                                self.cfg.decoarse.emb_features, self.sign_net,
                                                                self.spectrum_extractor,
                                                                eval_model)
                    pred_graphs.extend(pred_graphs_batch)

                sampled_data = utils.batch_sparse_tensors(pred_graphs, node_feature_dim=self.Xdim_decoarse,
                                                          edge_feature_dim=self.Edim_decoarse)
                sampled_dense_data, sampled_node_mask = utils.to_dense(sampled_data.x, sampled_data.edge_index,
                                                                       sampled_data.edge_attr, sampled_data.batch)
                sampled_dense_data = sampled_dense_data.mask(sampled_node_mask)

                all_losses = [graph_data['loss'] for graph_data in outputs]
                avg_loss = torch.stack(all_losses).mean().item()
                if wandb.run:
                    wandb.log({"val/loss": avg_loss,
                               "val/epoch_NLL": self.val_nll}, commit=False)

                pred_X_prob = F.softmax(sampled_dense_data.X, dim=-1)
                pred_E_prob = F.softmax(sampled_dense_data.E, dim=-1)

                # Convert to discrete predictions
                X = torch.argmax(pred_X_prob, dim=-1)
                E = torch.argmax(pred_E_prob, dim=-1)

                batch_graph = X.shape[0]
                target_n_nodes_all = torch.cat(all_target_n_nodes, dim=0)

                molecule_list = []
                for i in range(batch_graph):
                    n = int(target_n_nodes_all[i])
                    atom_types = X[i, :n].cpu()
                    edge_types = E[i, :n, :n].cpu()
                    molecule_list.append([atom_types, edge_types])

                if self.visualization_tools is not None:
                    self.print("Visualizing molecules...")

                    current_path = os.getcwd()
                    result_path = os.path.join(current_path, f'graphs/{self.name}/epoch{self.current_epoch}_validation/')
                    if not os.path.exists(result_path):
                        os.makedirs(result_path)

                    # Visualize molecules and save results
                    self.visualization_tools.visualize(result_path, molecule_list, self.cfg.general.samples_to_save)

                    self.print(f"Visualization complete. Graphs saved to: {result_path}")

                # self.validation_outputs = []
                self.validation_outputs.clear()
        else:
            metrics = [self.val_nll.compute(), self.val_X_kl.compute() * self.T, self.val_E_kl.compute() * self.T,
                       self.val_X_logp.compute(), self.val_E_logp.compute(), self.val_decoarsen_nll.compute()]

            total_val_loss = metrics[0] + metrics[5]

            if wandb.run:
                wandb.log({"val/epoch_diffusion_NLL": metrics[0],
                           "val/X_kl": metrics[1],
                           "val/E_kl": metrics[2],
                           "val/X_logp": metrics[3],
                           "val/E_logp": metrics[4],
                           "val/epoch_decoarse_NLL": metrics[5],
                           "val/epoch_NLL": total_val_loss,}, commit=False)

            self.print(f"Epoch {self.current_epoch}: Val loss {total_val_loss:.2f} "
                       f"-- Diffusion loss: {metrics[0]:.2f} -- Decoarsening loss: {metrics[5]:.2f}")

            self.log("val/epoch_NLL", total_val_loss, sync_dist=True)

            if total_val_loss < self.best_val_nll:
                self.best_val_nll = total_val_loss
            self.print('Val loss: %.4f \t Best val loss:  %.4f\n' % (total_val_loss, self.best_val_nll))

            # Start sampling/generating graphs for diffusion and augmenting the graphs
            self.val_counter += 1
            if self.val_counter % self.cfg.general.sample_every_val == 0:
                start = time.time()
                samples_left_to_generate = self.cfg.general.samples_to_generate
                samples_left_to_save = self.cfg.general.samples_to_save
                chains_left_to_save = self.cfg.general.chains_to_save

                molecule_list = []
                molecule_list_coarse = []

                ident = 0
                while samples_left_to_generate > 0:
                    bs = 2 * self.cfg.train.batch_size
                    to_generate = min(samples_left_to_generate, bs)
                    to_save = min(samples_left_to_save, bs)
                    chains_save = min(chains_left_to_save, bs)
                    molecule_list_new, molecule_list_coarse_new = self.sample_batch(
                                                        batch_id=ident, batch_size=to_generate,
                                                        num_nodes=None, save_final=to_save,
                                                        keep_chain=chains_save,
                                                        number_chain_steps=self.number_chain_steps)
                    molecule_list.extend(molecule_list_new)
                    molecule_list_coarse.extend(molecule_list_coarse_new)
                    ident += to_generate

                    samples_left_to_save -= to_save
                    samples_left_to_generate -= to_generate
                    chains_left_to_save -= chains_save

        to_log = {}
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            self.print("Computing sampling metrics for coarse...")
            to_log = self.sampling_metrics(molecule_list_coarse, 'diffusion', self.current_epoch, val_counter=-1,
                                           test=False,
                                           local_rank=self.local_rank,
                                           coarse=True)

            self.print("Computing sampling metrics for expansion...")
            to_log = self.sampling_metrics(molecule_list, 'expansion', self.current_epoch, val_counter=-1, test=False,
                                          local_rank=self.local_rank)
            self.print(f'Done. Sampling took {time.time() - start:.2f} seconds\n')
            print("Validation epoch end ends...")

        default = torch.tensor(0.0, device=self.device)
        vun = to_log.get('VUN', default)
        self.log(
            "val/gen_accuracy",
            vun,
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, data, i):
        if self.cfg.train.pretrain_decoarsen:
            # Pretraining Decoarsening Module
            eval_model = self.decoarsen_model_ema if (self.ema and not self.training) else self.decoarsen_model
            loss_decoarse, loss_terms = self.get_loss_decoarsen(data.adj_reduced, data, eval_model, self.sign_net)
            loss_c = 0.0
            self.test_decoarsen_nll(loss_decoarse)
        else:
            # Full Pipeline Training (Diffusion + Decoarsening)
            # Compute diffusion loss on the coarsened graph
            pred_c, dense_data_c, node_mask = self.get_pred(data.reduced_node_expansion, data.adj_reduced,
                                                            data.batch_reduced)

            loss_c = self.train_loss(masked_pred_X=pred_c.X, masked_pred_E=pred_c.E, pred_y=pred_c.y,
                                     true_X=dense_data_c.X, true_E=dense_data_c.E, true_y=data.y,
                                     log=False)

            eval_model = self.decoarsen_model_ema if (self.ema and not self.training) else self.decoarsen_model
            loss_decoarse, loss_terms = self.get_loss_decoarsen(data.adj_reduced, data, eval_model, self.sign_net)
            self.test_nll(loss_c)
            self.test_decoarsen_nll(loss_decoarse)

        total_loss = loss_c + loss_decoarse

        if wandb.run:
            wandb.log({
                "test/diffusion_loss": loss_c,
                "test/decoarsening_loss": loss_decoarse,
                "test/total_loss": total_loss,
            }, commit=False)

        if self.cfg.train.pretrain_decoarsen:
            batch_size, _, _ = dense_data.X.shape
            output = {"dense_data": dense_data, "node_mask": node_mask,
                      "batch_size": batch_size, 'loss': total_loss}
            self.test_outputs.append(output)

        return {'loss': total_loss, 'diffusion_loss': loss_c, 'decoarsening_loss': loss_decoarse}


    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_E_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
        self.test_decoarsen_nll.reset()
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)


    def on_test_epoch_end(self) -> None:
        if self.cfg.train.pretrain_decoarsen:
            decoarse_nll = self.test_decoarsen_nll.compute()
            if wandb.run:
                wandb.log({"test/epoch_decoarse_NLL": decoarse_nll}, commit=False)
            self.print(f"Epoch {self.current_epoch}: Test Decoarsen NLL: {decoarse_nll:.2f}")
            self.log("test/epoch_NLL", decoarse_nll, sync_dist=True)

            pred_graphs = []
            all_target_n_nodes = []
            outputs = self.test_outputs
            print('Start sampling...')
            for graph_data in outputs:
                dense_data = graph_data["dense_data"]
                node_mask = graph_data["node_mask"]
                batch_size = graph_data["batch_size"]
                target_n_nodes = self.orig_node_dist.sample_n(batch_size, self.device)
                all_target_n_nodes.append(target_n_nodes)
                eval_model = self.decoarsen_model_ema if (self.ema and not self.training) else self.decoarsen_model

                # Convert to discrete predictions
                discrete_X = torch.argmax(dense_data.X, dim=-1)
                discrete_E = torch.argmax(dense_data.E, dim=-1)

                # Compute the number of valid nodes for each graph
                n_nodes = [int(mask.sum().item()) for mask in node_mask]

                molecule_list = []
                batch_size = discrete_X.shape[0]
                for idx in range(batch_size):
                    n = n_nodes[idx]
                    atom_types = discrete_X[idx, :n].cpu()
                    edge_types = discrete_E[idx, :n, :n].cpu()
                    molecule_list.append([atom_types, edge_types])

                if self.visualization_tools is not None:
                    self.print("Visualizing molecules...")

                    current_path = os.getcwd()
                    result_path = os.path.join(current_path,
                                               f'graphs/{self.name}/epoch{self.current_epoch}_validation/')
                    if not os.path.exists(result_path):
                        os.makedirs(result_path)

                    # Visualize molecules and save results
                    self.visualization_tools.visualize(result_path, molecule_list, 1)

                    self.print(f"Visualization complete. Graphs saved to: {result_path}")
                breakpoint()

                pred_graphs_batch = self.sample_pred_graphs(dense_data.X, dense_data.E, node_mask,
                                                            target_n_nodes, self.decoarsen_diffusion,
                                                            self.cfg.decoarse.emb_features, self.sign_net,
                                                            self.spectrum_extractor,
                                                            eval_model)
                                                            # self.decoarsen_model)
                pred_graphs.extend(pred_graphs_batch)

            sampled_data = utils.batch_sparse_tensors(pred_graphs, node_feature_dim=self.Xdim_decoarse,
                                                      edge_feature_dim=self.Edim_decoarse)
            sampled_dense_data, sampled_node_mask = utils.to_dense(sampled_data.x, sampled_data.edge_index,
                                                                   sampled_data.edge_attr, sampled_data.batch)
            sampled_dense_data = sampled_dense_data.mask(sampled_node_mask)

            all_losses = [graph_data['loss'] for graph_data in outputs]
            avg_loss = torch.stack(all_losses).mean().item()
            if wandb.run:
                wandb.log({"test/loss": avg_loss,
                           "test/epoch_NLL": self.test_nll}, commit=False)

            pred_X_prob = F.softmax(sampled_dense_data.X, dim=-1)
            pred_E_prob = F.softmax(sampled_dense_data.E, dim=-1)

            # Convert to discrete predictions
            X = torch.argmax(pred_X_prob, dim=-1)
            E = torch.argmax(pred_E_prob, dim=-1)

            batch_graph = X.shape[0]
            target_n_nodes_all = torch.cat(all_target_n_nodes, dim=0)

            molecule_list = []
            for i in range(batch_graph):
                n = int(target_n_nodes_all[i])
                atom_types = X[i, :n].cpu()
                edge_types = E[i, :n, :n].cpu()
                molecule_list.append([atom_types, edge_types])

            if self.visualization_tools is not None:
                self.print("Visualizing molecules...")

                current_path = os.getcwd()
                result_path = os.path.join(current_path, f'graphs/{self.name}/epoch{self.current_epoch}_validation/')
                if not os.path.exists(result_path):
                    os.makedirs(result_path)

                # Visualize molecules and save results
                self.visualization_tools.visualize(result_path, molecule_list, self.cfg.general.samples_to_save)

                self.print(f"Visualization complete. Graphs saved to: {result_path}")

            self.test_outputs.clear()
        else:
            metrics = [self.test_nll.compute(), self.test_X_kl.compute() * self.T, self.test_E_kl.compute() * self.T,
                       self.test_X_logp.compute(), self.test_E_logp.compute(), self.test_decoarsen_nll.compute()]

            total_test_loss = metrics[0] + metrics[5]

            if wandb.run:
                wandb.log({"test/epoch_diffusion_NLL": metrics[0],
                           "test/X_kl": metrics[1],
                           "test/E_kl": metrics[2],
                           "test/X_logp": metrics[3],
                           "test/E_logp": metrics[4],
                           "test/epoch_decoarse_NLL": metrics[5],
                           "test/epoch_NLL": total_test_loss,}, commit=False)

            self.print(f"Epoch {self.current_epoch}: Test loss {total_test_loss:.2f} "
                       f"-- Diffusion loss: {metrics[0]:.2f} -- Decoarsening loss: {metrics[5]:.2f}")

            # Log the total validation loss using Lightning’s logger.
            self.log("test/epoch_NLL", total_test_loss, sync_dist=True)

            self.print(f'Test loss: {total_test_loss :.4f}')

            samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
            samples_left_to_save = self.cfg.general.final_model_samples_to_save
            chains_left_to_save = self.cfg.general.final_model_chains_to_save

            molecule_list = []
            molecule_list_coarse = []
            id = 0
            while samples_left_to_generate > 0:
                self.print(f'Samples left to generate: {samples_left_to_generate}/'
                           f'{self.cfg.general.final_model_samples_to_generate}', end='', flush=True)
                bs = 2 * self.cfg.train.batch_size
                to_generate = min(samples_left_to_generate, bs)
                to_save = min(samples_left_to_save, bs)
                chains_save = min(chains_left_to_save, bs)
                molecule_list_new, molecule_list_coarse_new = self.sample_batch(id, to_generate,
                                                                                num_nodes=None, save_final=to_save,
                                                                                keep_chain=chains_save,
                                                                                number_chain_steps=self.number_chain_steps)
                molecule_list.extend(molecule_list_new)
                molecule_list_coarse.extend(molecule_list_coarse_new)

                id += to_generate
                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save
        self.print("Saving the generated graphs")
        filename = f'generated_samples1.txt'
        for i in range(2, 10):
            if os.path.exists(filename):
                filename = f'generated_samples{i}.txt'
            else:
                break
        with open(filename, 'w') as f:
            for item in molecule_list:
                f.write(f"N={item[0].shape[0]}\n")
                atoms = item[0].tolist()
                f.write("X: \n")
                for at in atoms:
                    f.write(f"{at} ")
                f.write("\n")
                f.write("E: \n")
                for bond_list in item[1]:
                    for bond in bond_list:
                        f.write(f"{bond} ")
                    f.write("\n")
                f.write("\n")
        self.print("Generated graphs Saved.")

        self.print("Computing sampling metrics for coarse...")
        self.sampling_metrics(molecule_list_coarse, 'diffusion', self.current_epoch, val_counter=-1,
                                       test=True,
                                       local_rank=self.local_rank, coarse=True)

        self.print("Computing sampling metrics for expansion...")
        self.sampling_metrics(molecule_list, 'expansion', self.current_epoch, val_counter=-1, test=True,
                                       local_rank=self.local_rank)
        self.print("Done testing.")


    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        if self.cfg.model.uncon_gen is False:
            y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
            return self.model(X, E, y, node_mask)
        else:
            return self.model(X, E, node_mask)

    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.coarse_node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            write_index = (s_int * number_chain_steps) // self.T
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=False)
        pred_X_prob = F.softmax(sampled_s.X.float(), dim=-1)
        pred_E_prob = F.softmax(sampled_s.E.float(), dim=-1)

        # Convert to discrete predictions
        X = torch.argmax(pred_X_prob, dim=-1)
        E = torch.argmax(pred_E_prob, dim=-1)

        molecule_list_coarse = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list_coarse.append([atom_types, edge_types])

        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
        E = E.float()

        target_n_nodes = self.orig_node_dist.sample_n(batch_size, self.device)
        eval_model = self.decoarsen_model_ema if (self.ema and not self.training) else self.decoarsen_model
        pred_graphs = self.sample_pred_graphs(X, E, node_mask, target_n_nodes,
                                              self.decoarsen_diffusion,
                                              self.cfg.decoarse.emb_features,
                                              self.sign_net,
                                              self.spectrum_extractor,
                                              eval_model)

        sampled_data = utils.batch_sparse_tensors(pred_graphs, node_feature_dim=self.Xdim_decoarse,
                                                  edge_feature_dim=self.Edim_decoarse)
        sampled_dense_data, sampled_node_mask = utils.to_dense(sampled_data.x, sampled_data.edge_index,
                                            sampled_data.edge_attr, sampled_data.batch)
        sampled_dense_data = sampled_dense_data.mask(sampled_node_mask)

        pred_X_prob = F.softmax(sampled_dense_data.X, dim=-1)
        pred_E_prob = F.softmax(sampled_dense_data.E, dim=-1)

        # Convert to discrete predictions
        X = torch.argmax(pred_X_prob, dim=-1)
        E = torch.argmax(pred_E_prob, dim=-1)

        molecule_list = []
        for i in range(batch_size):
            n = target_n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        # Visualize molecules
        if self.visualization_tools is not None:
            self.print('\nVisualizing coarse molecules...')

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/diffusion/epoch{self.current_epoch}_b{batch_id}/')
            self.visualization_tools.visualize(result_path, molecule_list_coarse, save_final)
            self.print("Done.")

            self.print('\nVisualizing expanded molecules...')
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/expansion/epoch{self.current_epoch}_b{batch_id}/')
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list, molecule_list_coarse

    @torch.no_grad()
    def sample_pred_graphs(self, X, E, node_mask, target_n_nodes, decoarsen_diffusion, emb_features,
                           sign_net, spectrum_extractor, model):
        E_sparse = utils.batched_dense_to_sparse(E, node_mask)
        bs, n_c, feature = X.shape
        batch_reduced = (torch.arange(bs).unsqueeze(1).expand(bs, n_c).reshape(-1)).to(self.device)

        extra = target_n_nodes - n_c
        alpha = torch.ones(bs, n_c, device=self.device)
        proportions = torch.distributions.Dirichlet(alpha).rsample()

        allocation = torch.floor(proportions * extra[:, None]).to(torch.int)
        node_expansion = allocation + 1

        current_sum = node_expansion.sum(dim=1)
        diff = target_n_nodes - current_sum

        for i in range(bs):
            if diff[i] > 0:
                indices = torch.randperm(n_c, device=self.device)[:diff[i].item()]
                node_expansion[i, indices] += 1
            elif diff[i] < 0:
                indices = torch.randperm(n_c, device=self.device)[:(-diff[i]).item()]
                node_expansion[i, indices] -= 1

        node_expansion = node_expansion.reshape(-1)

        pred_graphs_batch = self.expansion.sample_graphs_with_initial(
            adj=E_sparse,
            batch=batch_reduced,
            node_expansion=node_expansion,
            target_size=target_n_nodes,
            model=model,
            sign_net=sign_net,
            spectrum_extractor=spectrum_extractor,
            decoarsen_diffusion=decoarsen_diffusion,
            emb_features=emb_features,
            iterative=self.cfg.reduction.iterative
        )
        return pred_graphs_batch