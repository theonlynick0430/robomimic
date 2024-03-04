from bc_benchmark_algos.dataset.dataset import MIMO_Dataset
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import RolloutPolicy
import torch
import imageio
import time
import json
import numpy as np
import os


class RolloutEnv:
    """
    Abstract class used to rollout policies in different environments. 
    """

    def __init__(self, config, validset):
        """
        Args:
            config (BaseConfig instance): config object

            validset (Dataset instance): validation dataset
        """
        self.config = config
        self.validset = validset

        assert isinstance(self.validset, MIMO_Dataset)
        assert self.validset.pad_frame_stack and self.validset.pad_seq_length, "validation set must pad frame stack and seq"

        self.create_env()

    def create_env(self):
        """
        Create environment associated with dataset.
        """
        return NotImplementedError
    
    def inputs_from_initial_obs(self, obs, demo_id):
        """
        Create inputs for model from initial observation by repeating it.

        Args: 
            obs (dict): maps obs_key to data of shape [D]

            demo_id (str): id of the demo, e.g., demo_0

        Returns:
            x (dict): maps obs_group to obs_key to
                np.array of shape [B=1, T=pad_frame_stack+1, D]
        """
        x = dict()
        # populate input with first obs and goal
        if len(ObsUtils.OBS_GROUP_TO_KEYS["obs"]) > 0:
            x["obs"] = dict()
        for obs_key in ObsUtils.OBS_GROUP_TO_KEYS["obs"]:
            assert obs_key in obs, f"could not find obs_key {obs_key} in obs from environment"
            x["obs"][obs_key] = obs[obs_key]
        # add batch, seq dim
        x = TensorUtils.to_batch(x)
        x = TensorUtils.to_sequence(x)
        # repeat along seq dim n_frame_stack+1 times to prepare history
        x = TensorUtils.repeat_seq(x=x, k=self.validset.n_frame_stack+1)
        # fetch initial goal
        x["goal"] = self.fetch_goal(demo_id=demo_id, t=0)
        return x
    
    def inputs_from_new_obs(self, x, obs, demo_id, t):
        """
        Update inputs for model by shifting history and inserting new observation.

        Args: 
            x (dict): maps obs_group to obs_key to
              np.array of shape [B=1, T=pad_frame_stack+1, D]

            obs (dict): maps obs_key to data of shape [D]

            demo_id (str): id of the demo, e.g., demo_0

            t (int): timestep in trajectory

        Returns:
            updated input @x
        """
        # update input using new obs
        x = TensorUtils.shift_seq(x=x, k=-1)
        for obs_key in ObsUtils.OBS_GROUP_TO_KEYS["obs"]:
            assert obs_key in obs, f"could not find obs_key {obs_key} in obs from environment"
            # only update last seq index to preserve history
            x["obs"][obs_key][:, -1, :] = obs[obs_key]
        # fetch new goal
        x["goal"] = self.fetch_goal(demo_id=demo_id, t=t)
        return x
    
    def fetch_goal(self, demo_id, t):
        """
        Get goal for specified demo and time.

        Args: 
            demo_id (str): id of the demo, e.g., demo_0

            t (int): timestep in trajectory

        Returns:
            goal seq np.array of shape [B=1, T=validset.n_frame_stack+1, D]
        """
        return NotImplementedError

    def run_rollout(
            self, 
            policy, 
            demo_id,
            video_writer=None,
            video_skip=5,
            terminate_on_success=False,
        ):
        """
        Run rollout on a single demo and save stats (and video if necessary).

        Args:
            policy (RolloutPolicy instance): policy to use for rollouts

            demo_id (str): id of demo to rollout

            video_writer (imageio Writer instance): if not None, use video writer object to append frames at 
                rate given by @video_skip

            video_skip (int): how often to write video frame

            terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

        Returns:
            results (dict): dictionary of results with the keys 
            "Return", "Horizon", "Success_Rate", "{metric}_Success_Rate"
        """
        assert isinstance(policy, RolloutPolicy)

    def rollout_with_stats(
            self, 
            policy, 
            demo_id,
            video_dir=None,
            video_writer=None,
            video_skip=5,
            terminate_on_success=False, 
            verbose=False,
        ):        
        """
        Configure video writer, run rollout, and log progress. 

        Args:
            policy (RolloutPolicy instance): policy to use for rollouts

            demo_id (str): id of demo to rollout

            video_dir (str): (optional) directory to save rollout videos

            video_writer (imageio Writer instance): if not None, use video writer object to append frames at 
                rate given by @video_skip

            video_skip (int): how often to write video frame

            terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

            verbose (bool): if True, print results of each rollout

        Returns:
            results (dict): dictionary of results with the keys 
            "Time", "Return", "Horizon", "Success_Rate", "{metric}_Success_Rate"
        """
        assert isinstance(policy, RolloutPolicy)

        rollout_timestamp = time.time()

        # create video writer
        write_video = video_dir is not None
        video_path = None
        if write_video and video_writer is None:
            video_str = f"{demo_id}.mp4"
            video_path = os.path.join(video_dir, f"{video_str}")
            video_writer = imageio.get_writer(video_path, fps=20)
            print("video writes to " + video_path)
        
        rollout_info = self.run_rollout(
            policy=policy, 
            demo_id=demo_id, 
            video_writer=video_writer, 
            video_skip=video_skip, 
            terminate_on_success=terminate_on_success, 
        )
        rollout_info["Time"] = time.time() - rollout_timestamp
        if verbose:
            horizon = rollout_info["Horizon"]
            success = rollout_info["Success_Rate"]
            print(f"demo={demo_id}, horizon={horizon}, success={success}")
            print(json.dumps(rollout_info, sort_keys=True, indent=4))

        if write_video:
            video_writer.close()

        return rollout_info
