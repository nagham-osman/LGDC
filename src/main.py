import graph_tool as gt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pathlib
import warnings
import socket

import torch
torch.cuda.empty_cache()
import torch.distributed as dist
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics

from diffusion_model import LiftedDenoisingDiffusion
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from diffusion_model_coarsen import CoarsenedDDM
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures

warnings.filterwarnings("ignore", category=PossibleUserWarning)

_real_torch_load = torch.load
def _load_force_full_weights(f, *args, **kwargs):
    kwargs.pop('weights_only', None)
    return _real_torch_load(f, *args, weights_only=False, **kwargs)
torch.load = _load_force_full_weights

def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only

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

    real_torch_load = torch.load
    torch.load = lambda f, **kwargs: real_torch_load(f, **{**kwargs, 'weights_only': kwargs.get('weights_only', False)})
    if cfg.model.type == 'discrete' and not cfg.reduction.coarsening:
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    elif cfg.model.type == 'discrete' and cfg.reduction.coarsening:
        model = CoarsenedDDM.load_from_checkpoint(resume, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    torch.load = real_torch_load
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]

    resume_path = os.path.join(root_dir, cfg.general.resume)

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
        # dist.init_process_group(
        #     backend="gloo",
        #     init_method="tcp://127.0.0.1:29501",
        #     rank=0,
        #     world_size=1
        # )

    real_torch_load = torch.load
    torch.load = lambda f, **kwargs: real_torch_load(f, **{**kwargs, 'weights_only': kwargs.get('weights_only', False)})
    if cfg.model.type == 'discrete' and not cfg.reduction.coarsening:
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    elif cfg.model.type == 'discrete' and cfg.reduction.coarsening:
        model = CoarsenedDDM.load_from_checkpoint(resume_path, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)

    torch.load = real_torch_load
    new_cfg = model.cfg

    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '_resume'

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model



@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]

    if dataset_config["name"] in ['planar', 'sbm', 'sbm_pos', 'sbm_neg',
                                  'comm20', 'comm_pos', 'comm_neg', 'coarsen_sbm',
                                  'coarsen_comm_pos', 'coarsen_comm20', 'coarsen_planar', 'coarsen_tree']:
        from datasets.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos
        from datasets.coarsen_spectre_dataset import CoarsenedSpectreGraphDataModule, CoarsenedSpectreDatasetInfos
        from analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics, TreeSamplingMetrics
        from analysis.visualization import NonMolecularVisualization

        if dataset_config['name'] in ['coarsen_sbm', 'coarsen_comm_pos', 'coarsen_comm20', 'coarsen_planar', 'coarsen_tree']:
            datamodule = CoarsenedSpectreGraphDataModule(cfg)
            print("getting infos")
            dataset_infos = CoarsenedSpectreDatasetInfos(datamodule, dataset_config)
        else:
            datamodule = SpectreGraphDataModule(cfg)
            dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)

        if dataset_config['name'] in ['sbm', 'sbm_pos', 'sbm_neg', 'coarsen_sbm']:
            sampling_metrics = SBMSamplingMetrics(datamodule)
        elif dataset_config['name'] in ['comm20', 'comm_pos', 'comm_neg', 'coarsen_comm_pos', 'coarsen_comm20']:
            sampling_metrics = Comm20SamplingMetrics(datamodule)
        elif dataset_config['name'] in ['coarsen_tree', 'tree']:
            sampling_metrics = TreeSamplingMetrics(datamodule)
        else:
            sampling_metrics = PlanarSamplingMetrics(datamodule)


        train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
        visualization_tools = NonMolecularVisualization()

        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}

    elif dataset_config["name"] in ['qm9', 'guacamol', 'moses']:
        from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
        from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
        from diffusion.extra_features_molecular import ExtraMolecularFeatures
        from analysis.visualization import MolecularVisualization

        if dataset_config["name"] == 'qm9':
            from datasets import qm9_dataset
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            train_smiles = qm9_dataset.get_train_smiles(cfg=cfg, train_dataloader=datamodule.train_dataloader(),
                                                        dataset_infos=dataset_infos, evaluate_dataset=False)
        elif dataset_config['name'] == 'guacamol':
            from datasets import guacamol_dataset
            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
            train_smiles = None

        elif dataset_config.name == 'moses':
            from datasets import moses_dataset
            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
            train_smiles = None
        else:
            raise ValueError("Dataset not implemented")

        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        if cfg.model.type == 'discrete':
            train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
        else:
            train_metrics = TrainMolecularMetrics(dataset_infos)

        # We do not evaluate novelty during training
        sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
        visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    if cfg.general.test_only:
        # When testing, previous configuration is fully loaded
        cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split('checkpoints')[0])

    utils.create_folders(cfg)

    if cfg.model.type == 'discrete' and not cfg.reduction.coarsening:
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
    elif cfg.model.type == 'discrete' and cfg.reduction.coarsening:
        model = CoarsenedDDM(cfg=cfg, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion(cfg=cfg, **model_kwargs)

    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

        acc_ckpt = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}",
            filename="acc-{epoch}",
            monitor="val/gen_accuracy",
            save_top_k=5,
            mode="max"
        )
        callbacks.append(acc_ckpt)

    # if cfg.train.ema_decay > 0:
    #     ema_callback = utils.EMA(beta=cfg.train.ema_decay)
        # callbacks.append(ema_callback)

    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
                      accelerator='gpu' if use_gpu else 'cpu',
                      devices=cfg.general.gpu_id if use_gpu else 1,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=False,
                      callbacks=callbacks,
                      log_every_n_steps=50 if name != 'debug' else 1,
                      logger = [])

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.name not in ['debug', 'test']:
            trainer.test(model, datamodule=datamodule)
    else:
        # Start by evaluating test_only_path
        real_torch_load = torch.load
        torch.load = lambda f, **kwargs: real_torch_load(f, weights_only=False, **kwargs)
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        torch.load = real_torch_load
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if '.ckpt' in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
