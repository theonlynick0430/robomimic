from bc_benchmark_algos.dataset.dataset import MIMO_Dataset
from bc_benchmark_algos.dataset.robomimic import RobomimicDataset
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import RolloutPolicy
import torch


class RolloutEnv:
    """
    Abstract class used to rollout policies in different environments. 
    """

    def __init__(self, config, validset, video_dir=None):
        """
        Args:
            config (BaseConfig instance): config object

            validset (Dataset instance): validation dataset

            video_dir (str): (optional) directory to save rollout videos
        """
        self.config = config
        self.validset: RobomimicDataset = validset
        self.video_dir = video_dir
        self.device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

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
        Args: 
            obs (dict): maps obs_key to data of shape [D]

            demo_id (str): id of the demo, e.g., demo_0

        Returns:
            x (dict): maps obs_group to obs_key to data of shape [B=1, T=pad_frame_stack+1, D]
        """
        x = dict()
        # populate input with first obs and goal
        if len(ObsUtils.OBS_GROUP_TO_KEYS[self.config.algo_name]["obs"]) > 0:
            x["obs"] = dict()
        for obs_key in ObsUtils.OBS_GROUP_TO_KEYS[self.config.algo_name]["obs"]:
            assert obs_key in obs, f"could not find obs_key {obs_key} in obs from environment"
            x["obs"][obs_key] = obs[obs_key]
        # add batch, seq dim
        x = TensorUtils.to_tensor(x, device=self.device)
        x = TensorUtils.to_batch(x)
        x = TensorUtils.to_sequence(x)
        # repeat along seq dim n_frame_stack+1 times to prepare history
        x = TensorUtils.repeat_seq(x=x, k=self.validset.n_frame_stack+1)
        # fetch initial goal
        x["goal"] = self.fetch_goal(demo_id=demo_id, t=0)
        return ObsUtils.process_inputs(**x)
    
    def inputs_from_new_obs(self, x, obs, demo_id, t):
        """
        Args: 
            x (dict): maps obs_group to obs_key to data of form [B=1, T=pad_frame_stack+1, D]

            obs (dict): maps obs_key to data of form [D]

            demo_id (str): id of the demo, e.g., demo_0

            t (int): timestep in trajectory

        Returns:
            updated input @x
        """
        # update input using new obs
        x = TensorUtils.shift_seq(x=x, k=-1)
        for obs_key in ObsUtils.OBS_GROUP_TO_KEYS[self.config.algo_name]["obs"]:
            assert obs_key in obs, f"could not find obs_key {obs_key} in obs from environment"
            # only update last seq index to preserve history
            x["obs"][obs_key][:, -1, :] = ObsUtils.process_obs(obs=torch.Tensor(obs[obs_key], device=self.device), obs_key=obs_key)
        # fetch new goal
        x["goal"] = ObsUtils.process_obs_dict(obs_dict=self.fetch_goal(demo_id=demo_id, t=t))
        return x
    
    def fetch_goal(self, demo_id, t):
        """
        Args: 
            demo_id (str): id of the demo, e.g., demo_0

            t (int): timestep in trajectory

        Returns:
            goal seq tensor of shape [B=1, T=validset.n_frame_stack+1, D]
        """
        return NotImplementedError

    def run_rollout(
            self, 
            policy, 
            video_writer=None,
            video_skip=5,
            terminate_on_success=False,
        ):
        """
        Args:
            policy (RolloutPolicy instance): policy to use for rollouts

            video_writer (imageio Writer instance): if not None, use video writer object to append frames at 
                rate given by @video_skip

            video_skip (int): how often to write video frame

            terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered
        """
        assert isinstance(policy, RolloutPolicy)

    def rollout_with_stats(
            self, 
            policy, 
            video_writer=None,
            epoch=None,
            video_skip=5,
            terminate_on_success=False, 
            verbose=False,
        ):        
        """
        Args:
            policy (RolloutPolicy instance): policy to use for rollouts

            video_writer (imageio Writer instance): if not None, use video writer object to append frames at 
                rate given by @video_skip

            epoch (int): epoch number (used for video naming)

            video_skip (int): how often to write video frame

            terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

            verbose (bool): if True, print results of each rollout
        """
        assert isinstance(policy, RolloutPolicy)
