"""
This file contains abstract Dataset classes that are used by torch dataloaders
to fetch batches from datasets.
"""
import numpy as np
import torch.utils.data
import robomimic.utils.tensor_utils as TensorUtils


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
        self.index_to_demo_id = dict()  # index in total_num_sequences -> demo_id
        self.demo_id_to_start_index = dict()  # demo_id -> start index in total_num_sequences
        self.demo_id_to_demo_length = dict() # demo_id -> length of demo in data

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
            self.demo_id_to_start_index[demo_id] = self.total_num_sequences
            self.demo_id_to_demo_length[demo_id] = demo_length

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
                self.index_to_demo_id[self.total_num_sequences] = demo_id
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
        demo_id = self.index_to_demo_id[index]
        # convert index in total_num_sequences to index in data
        start_offset = 0 if self.pad_frame_stack else self.n_frame_stack
        demo_index = index - self.demo_id_to_start_index[demo_id] + start_offset

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
        demo_length = self.demo_id_to_demo_length[demo_id]
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
        demo_length = self.demo_id_to_demo_length[demo_id]
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