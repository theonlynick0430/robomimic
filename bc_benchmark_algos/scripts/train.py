import robomimic
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.log_utils as LogUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings
from bc_benchmark_algos.dataset.dataset import RobomimicDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import json
import os
import time
import argparse
import sys

def train(args):

    ### CONFIG ###

    print("\n============= New Training Run with Config =============")
    ext_cfg = json.load(open(args.config, 'r'))
    config = config_factory(ext_cfg["algo_name"])
    # update config with external json - this will throw errors if
    # the external config has keys not present in the base algo config
    with config.unlocked():
        config.update(ext_cfg)
    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
    config.train.data = args.dataset
    # send output to a temporary directory
    config.train.output_dir = args.output
    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()
    print(config)
    print("")

    # set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    torch.set_num_threads(2)


    ### OUTPUT ###

    log_dir, ckpt_dir, video_dir = TrainUtils.get_exp_dir(config)

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )


    ### OBS UTILS ###

    print("\n============= Observation Utils =============")
    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)
    # verify important obs utils
    print("OBS_MODALITIES_TO_KEYS:")
    print(ObsUtils.OBS_MODALITIES_TO_KEYS)
    print("OBS_KEYS_TO_MODALITIES:")
    print(ObsUtils.OBS_KEYS_TO_MODALITIES)
    print("OBS_SHAPES:")
    print(ObsUtils.OBS_SHAPES)
    print("OBS_GROUP_TO_KEYS:")
    print(ObsUtils.OBS_GROUP_TO_KEYS)
    print("")


    ### MODEL ####

    print("\n============= Model Summary =============")
    ac_dim = config.train.ac_dim
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=ObsUtils.OBS_SHAPES[config.algo_name],
        ac_dim=ac_dim,
        device=device,
    )
    print(model)  # print model summary
    print("")


    #### DATASET ###

    # make sure the dataset exists
    dataset_path = os.path.expanduser(config.train.data)
    if not os.path.exists(dataset_path):
        raise Exception("Dataset at provided path {} not found!".format(dataset_path))

    assert config.train.dataset_type == "robomimic", "only robomimic datasets currently supported"
    # config can contain an attribute to filter on
    train_filter_by_attribute = config.train.hdf5_filter_key
    valid_filter_by_attribute = config.train.hdf5_validation_filter_key
    if valid_filter_by_attribute is not None:
        assert config.experiment.validate, "specified validation filter key {}, but config.experiment.validate is not set".format(valid_filter_by_attribute)
    # load the dataset into memory
    if config.experiment.validate:
        assert not config.train.hdf5_normalize_obs, "no support for observation normalization with validation data yet"
        assert (train_filter_by_attribute is not None) and (valid_filter_by_attribute is not None), \
            "did not specify filter keys corresponding to train and valid split in dataset" \
            " - please fill config.train.hdf5_filter_key and config.train.hdf5_validation_filter_key"
        train_demo_keys = FileUtils.get_demos_for_filter_key(
            hdf5_path=os.path.expanduser(config.train.data),
            filter_key=train_filter_by_attribute,
        )
        valid_demo_keys = FileUtils.get_demos_for_filter_key(
            hdf5_path=os.path.expanduser(config.train.data),
            filter_key=valid_filter_by_attribute,
        )
        assert set(train_demo_keys).isdisjoint(set(valid_demo_keys)), "training demonstrations overlap with " \
            "validation demonstrations!"
        trainset = RobomimicDataset.dataset_factory(config=config, obs_group_to_keys=ObsUtils.OBS_GROUP_TO_KEYS[config.algo_name], filter_by_attribute=train_filter_by_attribute)
        validset = RobomimicDataset.dataset_factory(config=config, obs_group_to_keys=ObsUtils.OBS_GROUP_TO_KEYS[config.algo_name], filter_by_attribute=valid_filter_by_attribute)
    else:
        train_dataset = RobomimicDataset.dataset_factory(config=config, obs_group_to_keys=ObsUtils.OBS_GROUP_TO_KEYS[config.algo_name], filter_by_attribute=train_filter_by_attribute)
        validset = None
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")
    if validset is not None:
        valid_sampler = validset.get_dataset_sampler()
        print("\n============= Validation Dataset =============")
        print(validset)
        print("")

    # maybe retreve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True
    )
    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True
        )
    else:
        valid_loader = None


    # print all warnings before training begins
    print("*" * 50)
    print("Warnings generated by robomimic have been duplicated here (from above) for convenience. Please check them carefully.")
    flush_warnings()
    print("*" * 50)
    print("")


    ## TRAINING ###

    # main training loop
    last_ckpt_time = time.time()

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    for epoch in range(1, config.train.num_epochs + 1): # epoch numbers start at 1
        print(f"Epoch: {epoch}")
        step_log = TrainUtils.run_epoch(
            model=model,
            data_loader=train_loader,
            epoch=epoch,
            num_steps=train_num_steps,
            obs_normalization_stats=obs_normalization_stats,
        )
        model.on_epoch_end(epoch)

    # terminate logging
    data_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to a config json"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to dataset",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="path to output directory",
    )

    args = parser.parse_args()
    train(args)