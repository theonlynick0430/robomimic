"""
This file contains Dataset classes that are used by torch dataloaders
to fetch batches from hdf5 files.
"""
import os
import h5py
import numpy as np
from copy import deepcopy
from contextlib import contextmanager

import torch.utils.data

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.log_utils as LogUtils

class MIMO_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        obs_group_to_keys,
        dataset_keys,
        frame_stack=0,
        seq_length=1,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=False,
        goal_mode=None,
        num_subgoal=None,
    ):
        """
        Abstract class for fetching sequences of experience. Inherit from this class for 
        different dataset formats. 
        Length of the fetched sequence is equal to (@frame_stack + @seq_length)

        Args:
            obs_group_to_keys (dict(iterable)): dictionary that maps observation group (obs, goal etc) to 
              observation keys (image, proprio, etc) to be fetched from the dataset

            dataset_keys (tuple, list): keys to dataset items (actions, rewards, etc) to be fetched from the dataset.

            frame_stack (int): numbers of stacked frames to fetch. Defaults to 0 (single frame).

            seq_length (int): length of sequences to sample. Defaults to 1 (single frame).

            pad_frame_stack (int): whether to pad sequence for frame stacking at the beginning of a demo. This
                ensures that partial frame stacks are observed, such as (s_0, s_0, s_0, s_1). Otherwise, the
                first frame stacked observation would be (s_0, s_1, s_2, s_3).

            pad_seq_length (int): whether to pad sequence for sequence fetching at the end of a demo. This
                ensures that partial sequences at the end of a demonstration are observed, such as
                (s_{T-1}, s_{T}, s_{T}, s_{T}). Otherwise, the last sequence provided would be
                (s_{T-3}, s_{T-2}, s_{T-1}, s_{T}).

            get_pad_mask (bool): if True, also provide padding masks as part of the batch. This can be
                useful for masking loss functions on padded parts of the data.

            goal_mode (str): either "last", "subgoal", or None. Defaults to None, which is to not fetch goals

            num_subgoal (int): Required if goal_mode is "subgoal". Number of subgoals provided for each trajectory.
            Defaults to None, which indicates that every state is also a subgoal. Assume num_subgoal <= min length of traj.
        """
        self.obs_group_to_keys = obs_group_to_keys # obs group -> obs keys
        self.obs_keys = tuple(set([key for keys in self.obs_group_to_keys.values() for key in keys])) # obs keys for all obs groups (union)
        self.dataset_keys = tuple(dataset_keys) # obs keys for dataset

        self.n_frame_stack = frame_stack
        assert self.n_frame_stack >= 0
        self.seq_length = seq_length
        assert self.seq_length >= 1

        self.pad_seq_length = pad_seq_length
        self.pad_frame_stack = pad_frame_stack
        self.get_pad_mask = get_pad_mask

        self.goal_mode = goal_mode
        self.num_subgoal = num_subgoal
        if self.goal_mode is not None:
            assert self.goal_mode in ["last", "subgoal"]

        # init data structures 
        self.total_num_sequences = 0
        self._index_to_demo_id = dict()  # index in total_num_sequences -> demo_id
        self._demo_id_to_start_index = dict()  # demo_id -> start index in total_num_sequences
        self._demo_id_to_demo_length = dict() # demo_id -> length of demo in data

        self.load_demo_info()

    @classmethod
    def dataset_factory(cls, config, obs_group_to_keys):
        """
        Create a MIMO_Dataset instance from config.

        Args:
            config (BaseConfig instance): config object

            obs_group_to_keys (dict): dictionary from observation group to observation keys

        Returns:
            dataset (MIMO_Dataset instance): dataset object
        """
        ds_kwargs = dict(
            obs_group_to_keys=obs_group_to_keys,
            dataset_keys=config.train.dataset_keys,
            frame_stack=config.train.frame_stack,
            seq_length=config.train.seq_length,
            pad_frame_stack=config.train.pad_frame_stack,
            pad_seq_length=config.train.pad_seq_length,
            get_pad_mask=False,
            goal_mode=config.train.goal_mode,
            num_subgoal=config.train.num_subgoal
        )
        dataset = cls(**ds_kwargs)
        return dataset

    @property
    def demos(self):
        """
        Get all demo ids
        """
        return NotImplementedError      

    @property
    def num_demos(self):
        """
        Get number of demos
        """
        return NotImplementedError      

    def get_demo_len(self, demo_id):
        """
        Get length of demo with demo_id
        """
        return NotImplementedError

    def load_demo_info(self):
        """
        Populate internal data structures
        """
        for demo_id in self.demos:
            demo_length = self.get_demo_len(demo_id=demo_id)
            self._demo_id_to_start_index[demo_id] = self.total_num_sequences
            self._demo_id_to_demo_length[demo_id] = demo_length

            num_sequences = demo_length
            # determine actual number of sequences taking into account whether to pad for frame_stack and seq_length
            if not self.pad_frame_stack:
                num_sequences -= self.n_frame_stack
            if not self.pad_seq_length:
                num_sequences -= (self.seq_length - 1)

            if self.pad_seq_length:
                assert demo_length >= 1  # sequence needs to have at least one sample
                num_sequences = max(num_sequences, 1)
            else:
                assert num_sequences >= 1  # assume demo_length >= (self.n_frame_stack + self.seq_length)

            for _ in range(num_sequences):
                self._index_to_demo_id[self.total_num_sequences] = demo_id
                self.total_num_sequences += 1    

    def __len__(self):
        """
        Ensure that the torch dataloader will do a complete pass through all sequences in 
        the dataset before starting a new iteration.
        """
        return self.total_num_sequences
    
    def __getitem__(self, index):
        """
        Fetch dataset sequence @index (inferred through internal index map), using the getitem_cache if available.
        """
        return self.get_item(index)
    
    def __repr__(self):
        """
        Pretty print the class and important attributes on a call to `print`.
        """
        msg = "\tframe_stack={}\n\tseq_length={}\n"
        msg += "\tpad_seq_length={}\n\tpad_frame_stack={}\n\tgoal_mode={}\n\tnum_subgoal={}\n"
        msg += "\tnum_demos={}\n\tnum_sequences={}\n"
        goal_mode_str = self.goal_mode if self.goal_mode is not None else "none"
        num_subgoal_str = self.num_subgoal if self.num_subgoal is not None else "none"
        msg = msg.format(self.n_frame_stack, self.seq_length,
                         self.pad_seq_length, self.pad_frame_stack, goal_mode_str, num_subgoal_str,
                         self.num_demos, self.total_num_sequences)
        return msg
    
    def get_item(self, index):
        """
        Main implementation of getitem.
        """
        demo_id = self._index_to_demo_id[index]
        # convert index in total_num_sequences to index in data
        start_offset = 0 if self.pad_frame_stack else self.n_frame_stack
        demo_index = index - self._demo_id_to_start_index[demo_id] + start_offset

        data_seq_index, pad_mask = self.get_data_seq_index(demo_id=demo_id, index_in_demo=demo_index)
        meta = self.get_data_seq(demo_id=demo_id, keys=self.dataset_keys, seq_index=data_seq_index)
        meta["obs"] = self.get_obs_seq(demo_id=demo_id, keys=self.obs_group_to_keys["obs"], seq_index=data_seq_index)
        if "goal" in self.obs_group_to_keys and self.goal_mode in ["last", "subgoal"]:
            goal_index = self.get_goal_seq_index(demo_id=demo_id, data_seq_index=data_seq_index)
            meta["goal"] = self.get_obs_seq(demo_id=demo_id, keys=self.obs_group_to_keys["goal"], seq_index=goal_index)
        if self.get_pad_mask:
            meta["pad_mask"] = pad_mask

        return meta
    
    def get_data_seq(self, demo_id, keys, seq_index):
        """
        Extract a (sub)sequence of data items from a demo given the @keys of the items.

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            keys (tuple): list of keys to extract
            seq_index (list): sequence indices

        Returns:
            a dictionary of extracted items.
        """
        return NotImplementedError
    
    def get_obs_seq(self, demo_id, keys, seq_index):
        """
        Extract a (sub)sequence of observation items from a demo given the @keys of the items.

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            keys (tuple): list of keys to extract
            seq_index (list): sequence indices

        Returns:
            a dictionary of extracted items.
        """
        return NotImplementedError

    def get_data_seq_index(self, demo_id, index_in_demo):
        """
        Get sequence indices and pad mask to extract data from a demo. 

        Args:
            demo_id (key): id of the demo.
            index_in_demo (int): beginning index of the sequence wrt the demo.

        Returns:
            data sequence indices and pad mask.
        """
        demo_length = self._demo_id_to_demo_length[demo_id]
        assert index_in_demo < demo_length

        # determine begin and end of sequence
        seq_begin_index = max(0, index_in_demo - self.n_frame_stack)
        seq_end_index = min(demo_length, index_in_demo + self.seq_length)
        seq_index = np.arange(seq_begin_index, seq_end_index, dtype=np.int32)

        # determine sequence padding
        seq_begin_pad = max(0, self.n_frame_stack - index_in_demo)  # pad for frame stacking
        seq_end_pad = max(0, index_in_demo + self.seq_length - demo_length)  # pad for sequence length

        # make sure we are not padding if specified.
        if not self.pad_frame_stack:
            assert seq_begin_pad == 0
        if not self.pad_seq_length:
            assert seq_end_pad == 0

        seq_index = TensorUtils.pad_sequence(seq_index, padding=(seq_begin_pad, seq_end_pad))
        pad_mask = np.array([0] * seq_begin_pad + [1] * (seq_end_index - seq_begin_index) + [0] * seq_end_pad)
        pad_mask = pad_mask[:, None].astype(bool)
            
        return seq_index, pad_mask 
    
    def get_goal_seq_index(self, demo_id, data_seq_index):
        """
        Get sequence indices to extract goals from a demo. 

        Args:
            demo_id (key): id of the demo.
            data_seq_index (list): sequence indices.

        Returns:
            goal sequence indices.
        """
        demo_length = self._demo_id_to_demo_length[demo_id]
        goal_index = None
        if self.goal_mode == "last":
            goal_index = np.full((demo_length), -1) # every goal is the last state
        elif self.goal_mode == "subgoal" and self.num_subgoal is None:
            goal_index = np.arange(1, demo_length+1) # goal is the next state
        elif self.goal_mode == "subgoal":
            # create evenly spaced subgoals
            subgoal_index = np.linspace(0, demo_length, self.num_subgoal+1, dtype=int)
            repeat = np.diff(subgoal_index)
            goal_index = np.array([index for i, index in enumerate(subgoal_index[1:]) for _ in range(repeat[i])])

        goal_index = goal_index[data_seq_index]

        return goal_index

    def get_dataset_sampler(self):
        """
        Return instance of torch.utils.data.Sampler or None. Allows
        for dataset to define custom sampling logic, such as
        re-weighting the probability of samples being drawn.
        See the `train` function in scripts/train.py, and torch
        `DataLoader` documentation, for more info.
        """
        return None
    

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