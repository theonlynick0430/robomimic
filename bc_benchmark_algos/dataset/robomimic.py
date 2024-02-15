"""
This file contains Dataset classes that are used by torch dataloaders
to fetch batches from hdf5 files.
"""
from bc_benchmark_algos.dataset.dataset import MIMO_Dataset
import os
import h5py
import numpy as np
from copy import deepcopy
from contextlib import contextmanager
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.log_utils as LogUtils
import robomimic.utils.file_utils as FileUtils
from torch.utils.data import DataLoader


class RobomimicDataset(MIMO_Dataset):
    def __init__(
        self,
        hdf5_path,
        obs_group_to_keys,
        dataset_keys,
        frame_stack=0,
        seq_length=1,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=False,
        goal_mode=None,
        num_subgoal=None,
        hdf5_cache_mode=None,
        hdf5_use_swmr=True,
        hdf5_normalize_obs=False,
        filter_by_attribute=None,
    ):
        """
        MIMO_Dataset subclass for fetching sequences of experience from HDF5 dataset.

        Args:
            hdf5_path (str): path to hdf5.

            hdf5_cache_mode (str): one of ["all", "low_dim", or None]. Set to "all" to cache entire hdf5 
                in memory - this is by far the fastest for data loading. Set to "low_dim" to cache all 
                non-image data. Set to None to use no caching - in this case, every batch sample is 
                retrieved via file i/o. You should almost never set this to None, even for large 
                image datasets.

            hdf5_use_swmr (bool): whether to use swmr feature when opening the hdf5 file. This ensures
                that multiple Dataset instances can all access the same hdf5 file without problems.

            hdf5_normalize_obs (bool): if True, normalize observations by computing the mean observation
                and std of each observation (in each dimension and modality), and normalizing to unit
                mean and variance in each dimension.

            filter_by_attribute (str): if provided, use the provided filter key to look up a subset of
                demonstrations to load
        """
        self.hdf5_path = os.path.expanduser(hdf5_path)
        self.hdf5_use_swmr = hdf5_use_swmr
        self.hdf5_normalize_obs = hdf5_normalize_obs
        self._hdf5_file = None
        assert hdf5_cache_mode in ["all", "low_dim", None]
        self.hdf5_cache_mode = hdf5_cache_mode
        self.filter_by_attribute = filter_by_attribute

        super(RobomimicDataset, self).__init__(
            obs_group_to_keys=obs_group_to_keys,
            dataset_keys=dataset_keys,
            frame_stack=frame_stack,
            seq_length=seq_length, 
            pad_frame_stack=pad_frame_stack, 
            pad_seq_length=pad_seq_length, 
            get_pad_mask=get_pad_mask, 
            goal_mode=goal_mode, 
            num_subgoal=num_subgoal
            )
        
        # maybe prepare for observation normalization
        self.obs_normalization_stats = None
        if self.hdf5_normalize_obs:
            self.obs_normalization_stats = self.normalize_obs()

        # maybe store dataset in memory for fast access
        if self.hdf5_cache_mode in ["all", "low_dim"]:
            obs_keys_in_memory = self.obs_keys
            if self.hdf5_cache_mode == "low_dim":
                # only store low-dim observations
                obs_keys_in_memory = []
                for k in self.obs_keys:
                    if ObsUtils.key_is_obs_modality(k, "low_dim"):
                        obs_keys_in_memory.append(k)
            self.obs_keys_in_memory = obs_keys_in_memory

            self.hdf5_cache = self.load_dataset_in_memory(
                demo_list=self.demos,
                hdf5_file=self.hdf5_file,
                obs_keys=self.obs_keys_in_memory,
                dataset_keys=self.dataset_keys,
            )

            if self.hdf5_cache_mode == "all":
                # cache getitem calls for even more speedup. We don't do this for
                # "low-dim" since image observations require calls to getitem anyways.
                print("RobomimicDataset: caching get_item calls...")
                self.getitem_cache = [self.get_item(i) for i in LogUtils.custom_tqdm(range(len(self)))]

                # don't need the previous cache anymore
                del self.hdf5_cache
                self.hdf5_cache = None
        else:
            self.hdf5_cache = None

        self.close_and_delete_hdf5_handle()

    @classmethod
    def dataset_factory(cls, config, obs_group_to_keys, filter_by_attribute=None):
        """
        Create a RobomimicDataset instance from config.

        Args:
            config (BaseConfig instance): config object

            obs_group_to_keys (dict): dictionary from observation group to observation keys

            filter_by_attribute (str): if provided, use the provided filter key
                to select a subset of demonstration trajectories to load

        Returns:
            dataset (RobomimicDataset instance): dataset object
        """
        ds_kwargs = dict(
            hdf5_path=config.train.data,
            obs_group_to_keys=obs_group_to_keys,
            dataset_keys=config.train.dataset_keys,
            frame_stack=config.train.frame_stack,
            seq_length=config.train.seq_length,
            pad_frame_stack=config.train.pad_frame_stack,
            pad_seq_length=config.train.pad_seq_length,
            get_pad_mask=False,
            goal_mode=config.train.goal_mode,
            num_subgoal=config.train.num_subgoal,
            hdf5_cache_mode=config.train.hdf5_cache_mode,
            hdf5_use_swmr=config.train.hdf5_use_swmr,
            hdf5_normalize_obs=config.train.hdf5_normalize_obs,
            filter_by_attribute=filter_by_attribute
        )
        dataset = cls(**ds_kwargs)
        return dataset

    @property
    def demos(self):
        """
        Get all demo ids
        """
        demos = []
        # filter demo trajectory by mask
        if self.filter_by_attribute is not None:
            demos = [elem.decode("utf-8") for elem in np.array(self.hdf5_file["mask/{}".format(self.filter_by_attribute)][:])]
        else:
            demos = list(self.hdf5_file["data"].keys())
        # sort demo keys
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]
        return demos

    @property
    def num_demos(self):
        """
        Get number of demos
        """
        return len(self.demos)   

    def get_demo_len(self, demo_id):
        """
        Get length of demo with demo_id
        """
        return self.hdf5_file["data/{}".format(demo_id)].attrs["num_samples"] 

    @property
    def hdf5_file(self):
        """
        This property allows for a lazy hdf5 file open.
        """
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, 'r', swmr=self.hdf5_use_swmr, libver='latest')
        return self._hdf5_file  
    
    def close_and_delete_hdf5_handle(self):
        """
        Maybe close the file handle.
        """
        if self._hdf5_file is not None:
            self._hdf5_file.close()
        self._hdf5_file = None
    
    @contextmanager
    def hdf5_file_opened(self):
        """
        Convenient context manager to open the file on entering the scope
        and then close it on leaving.
        """
        should_close = self._hdf5_file is None
        yield self.hdf5_file
        if should_close:
            self.close_and_delete_hdf5_handle()
    
    def __del__(self):
        self.close_and_delete_hdf5_handle()

    def __repr__(self):
        """
        Pretty print the class and important attributes on a call to `print`.
        """
        msg = str(self.__class__.__name__)
        msg += " (\n\tpath={}\n\tobs_group_to_keys={}\n\tobs_keys={}\n\tfilter_key={}\n\tcache_mode={}\n"
        filter_key_str = self.filter_by_attribute if self.filter_by_attribute is not None else "none"
        cache_mode_str = self.hdf5_cache_mode if self.hdf5_cache_mode is not None else "none"
        msg = msg.format(self.hdf5_path, self.obs_group_to_keys, self.obs_keys, filter_key_str, cache_mode_str)
        return msg + super(RobomimicDataset, self).__repr__() + ")"

    def load_dataset_in_memory(self, demo_list, hdf5_file, obs_keys, dataset_keys):
        """
        Loads the hdf5 dataset into memory, preserving the structure of the file. Note that this
        differs from `self.getitem_cache`, which, if active, actually caches the outputs of the
        `getitem` operation.

        Args:
            demo_list (list): list of demo keys, e.g., 'demo_0'
            hdf5_file (h5py.File): file handle to the hdf5 dataset.
            obs_keys (list, tuple): observation keys to fetch, e.g., 'images'
            dataset_keys (list, tuple): dataset keys to fetch, e.g., 'actions'

        Returns:
            all_data (dict): dictionary of loaded data.
        """
        all_data = dict()
        print("RobomimicDataset: loading dataset into memory...")
        for ep in LogUtils.custom_tqdm(demo_list):
            all_data[ep] = {}
            all_data[ep]["attrs"] = {}
            all_data[ep]["attrs"]["num_samples"] = hdf5_file["data/{}".format(ep)].attrs["num_samples"]
            # get obs
            all_data[ep]["obs"] = {k: hdf5_file["data/{}/obs/{}".format(ep, k)][()] for k in obs_keys}
            # get other dataset keys
            for k in dataset_keys:
                if k in hdf5_file["data/{}".format(ep)]:
                    all_data[ep][k] = hdf5_file["data/{}/{}".format(ep, k)][()].astype('float32')
                else:
                    all_data[ep][k] = np.zeros((all_data[ep]["attrs"]["num_samples"], 1), dtype=np.float32)

            if "model_file" in hdf5_file["data/{}".format(ep)].attrs:
                all_data[ep]["attrs"]["model_file"] = hdf5_file["data/{}".format(ep)].attrs["model_file"]

        return all_data
    
    def normalize_obs(self):
        """
        Computes a dataset-wide mean and standard deviation for the observations 
        (per dimension and per obs key) and returns it.
        """
        def _compute_traj_stats(traj_obs_dict):
            """
            Helper function to compute statistics over a single trajectory of observations.
            """
            traj_stats = { k : {} for k in traj_obs_dict }
            for k in traj_obs_dict:
                traj_stats[k]["n"] = traj_obs_dict[k].shape[0]
                traj_stats[k]["mean"] = traj_obs_dict[k].mean(axis=0, keepdims=True) # [1, ...]
                traj_stats[k]["sqdiff"] = ((traj_obs_dict[k] - traj_stats[k]["mean"]) ** 2).sum(axis=0, keepdims=True) # [1, ...]
            return traj_stats

        def _aggregate_traj_stats(traj_stats_a, traj_stats_b):
            """
            Helper function to aggregate trajectory statistics.
            See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
            for more information.
            """
            merged_stats = {}
            for k in traj_stats_a:
                n_a, avg_a, M2_a = traj_stats_a[k]["n"], traj_stats_a[k]["mean"], traj_stats_a[k]["sqdiff"]
                n_b, avg_b, M2_b = traj_stats_b[k]["n"], traj_stats_b[k]["mean"], traj_stats_b[k]["sqdiff"]
                n = n_a + n_b
                mean = (n_a * avg_a + n_b * avg_b) / n
                delta = (avg_b - avg_a)
                M2 = M2_a + M2_b + (delta ** 2) * (n_a * n_b) / n
                merged_stats[k] = dict(n=n, mean=mean, sqdiff=M2)
            return merged_stats

        # Run through all trajectories. For each one, compute minimal observation statistics, and then aggregate
        # with the previous statistics.
        ep = self.demos[0]
        obs_traj = {k: self.hdf5_file["data/{}/obs/{}".format(ep, k)][()].astype('float32') for k in self.obs_keys}
        obs_traj = ObsUtils.process_obs_dict(obs_traj)
        merged_stats = _compute_traj_stats(obs_traj)
        print("RobomimicDataset: normalizing observations...")
        for ep in LogUtils.custom_tqdm(self.demos[1:]):
            obs_traj = {k: self.hdf5_file["data/{}/obs/{}".format(ep, k)][()].astype('float32') for k in self.obs_keys}
            obs_traj = ObsUtils.process_obs_dict(obs_traj)
            traj_stats = _compute_traj_stats(obs_traj)
            merged_stats = _aggregate_traj_stats(merged_stats, traj_stats)

        obs_normalization_stats = { k : {} for k in merged_stats }
        for k in merged_stats:
            # note we add a small tolerance of 1e-3 for std
            obs_normalization_stats[k]["mean"] = merged_stats[k]["mean"].astype(np.float32)
            obs_normalization_stats[k]["std"] = (np.sqrt(merged_stats[k]["sqdiff"] / merged_stats[k]["n"]) + 1e-3).astype(np.float32)
        return obs_normalization_stats
    
    def get_obs_normalization_stats(self):
        """
        Returns dictionary of mean and std for each observation key if using
        observation normalization, otherwise None.

        Returns:
            obs_normalization_stats (dict): a dictionary for observation
                normalization. This maps observation keys to dicts
                with a "mean" and "std" of shape (1, ...) where ... is the default
                shape for the observation.
        """
        assert self.hdf5_normalize_obs, "not using observation normalization!"
        return deepcopy(self.obs_normalization_stats)

    def get_dataset_for_ep(self, ep, key):
        """
        Helper utility to get a dataset for a specific demonstration.
        Takes into account whether the dataset has been loaded into memory.
        """

        # check if this key should be in memory
        key_should_be_in_memory = (self.hdf5_cache_mode in ["all", "low_dim"])
        if key_should_be_in_memory:
            # if key is an observation, it may not be in memory
            if '/' in key:
                key1, key2 = key.split('/')
                assert(key1 in ['obs'])
                if key2 not in self.obs_keys_in_memory:
                    key_should_be_in_memory = False

        if key_should_be_in_memory:
            # read cache
            if '/' in key:
                key1, key2 = key.split('/')
                assert(key1 in ['obs'])
                ret = self.hdf5_cache[ep][key1][key2]
            else:
                ret = self.hdf5_cache[ep][key]
        else:
            # read from file
            hd5key = "data/{}/{}".format(ep, key)
            ret = self.hdf5_file[hd5key]
        return ret

    def __getitem__(self, index):
        """
        Fetch dataset sequence @index (inferred through internal index map), using the getitem_cache if available.
        """
        if self.hdf5_cache_mode == "all":
            return self.getitem_cache[index]
        return super(RobomimicDataset, self).__getitem__(index=index)
    
    def get_data_seq(self, demo_id, keys, seq_index):
        """
        Extract a (sub)sequence of data items from a demo given the @keys of the items.

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            keys (tuple): list of keys to extract
            seq_index (tuple): sequence indices

        Returns:
            a dictionary of extracted items.
        """
        # fetch observation from the dataset file
        seq = dict()
        for k in keys:
            data = self.get_dataset_for_ep(demo_id, k)
            seq[k] = data[seq_index]

        return seq
    
    def get_obs_seq(self, demo_id, keys, seq_index):
        """
        Extract a (sub)sequence of observation items from a demo given the @keys of the items.

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            keys (tuple): list of keys to extract
            seq_index (tuple): sequence indices

        Returns:
            a dictionary of extracted items.
        """
        seq = self.get_data_seq(
            demo_id=demo_id,
            keys=['{}/{}'.format("obs", k) for k in keys], 
            seq_index=seq_index
            )
        seq = {k.split('/')[1]: seq[k] for k in seq}  # strip the prefix
        return seq
    

def load_train_val_data(config):
    """
    Main method for loading training and validation data from RobomimicDataset.

    Args:
        config (BaseConfig instance): config object

    Returns:
        trainset (RobomimicDataset instance), train_loader (DataLoader instance), 
        validset (RobomimicDataset instance), valid_loader (DataLoader instance)
    """
    trainset = None
    train_loader = None
    validset = None
    valid_loader = None

    # make sure the dataset exists
    dataset_path = os.path.expanduser(config.train.data)
    if not os.path.exists(dataset_path):
        raise Exception("Dataset at provided path {} not found!".format(dataset_path))
    
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
        trainset = RobomimicDataset.dataset_factory(config=config, obs_group_to_keys=ObsUtils.OBS_GROUP_TO_KEYS[config.algo_name], filter_by_attribute=train_filter_by_attribute)
    
    train_sampler = trainset.get_dataset_sampler()
    if validset is not None:
        valid_sampler = validset.get_dataset_sampler()

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

    return trainset, train_loader, validset, valid_loader