from bc_benchmark_algos.rollout_env.rollout_env import RolloutEnv
from bc_benchmark_algos.dataset.robomimic import RobomimicDataset
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.log_utils as LogUtils
from collections import OrderedDict
from robomimic.algo import algo_factory, RolloutPolicy
import imageio
import h5py
import numpy as np
import tqdm
import os
import time
import json


class RobomimicRolloutEnv(RolloutEnv):
    """
    Class used to rollout policies in in Robomimic environments. 
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

        assert isinstance(self.validset, RobomimicDataset)
        assert self.validset.pad_frame_stack and self.validset.pad_seq_length, "validation set must pad frame stack and seq"

        self.create_env()

    def inputs_from_initial_obs(self, obs):
        """
        Args: 
            obs (dict): maps obs_key to data of form [D]

        Returns:
            x (dict): maps obs_group to obs_key to data of form [B=1, T=pad_frame_stack+1, D]
        """
        x = dict()
        # populate input with first obs and goal
        x["obs"] = dict()
        for obs_key in ObsUtils.OBS_GROUP_TO_KEYS[self.config.algo_name]["obs"]:
            assert obs_key in obs, f"could not find obs_key {obs_key} in obs from environment"
            x["obs"][obs_key] = obs[obs_key]
        # add batch, seq dim
        x = TensorUtils.to_tensor(x)
        x = TensorUtils.to_batch(x)
        x = TensorUtils.to_sequence(x)
        # repeat along seq dim n_frame_stack+1 times to prepare history
        x = TensorUtils.repeat_seq(x=x, k=self.validset.n_frame_stack+1)
        # fetch initial goal
        x["goal"] = TensorUtils.to_batch(TensorUtils.to_tensor(self.fetch_goal(index=0)))
        return x
    
    def inputs_from_new_obs(self, x, obs, index):
        """
        Args: 
            x (dict): maps obs_group to obs_key to data of form [B=1, T=pad_frame_stack+1, D]

            obs (dict): maps obs_key to data of form [D]

            index (int): index of trajectory in dataset

        Returns:
            updated input @x
        """
        # update input using new obs
        x = TensorUtils.shift_seq(x=x, k=-1)
        for obs_key in ObsUtils.OBS_GROUP_TO_KEYS[self.config.algo_name]["obs"]:
            assert obs_key in obs, f"could not find obs_key {obs_key} in obs from environment"
            # only update last seq index to preserve history
            x["obs"][obs_key][:, -1, :] = obs[obs_key]
        # fetch new goal
        x["goal"] = TensorUtils.to_batch(TensorUtils.to_tensor(self.fetch_goal(index=index)))
        return x
    
    def fetch_goal(self, index):
        """
        Args: 
            index (int): index of trajectory in dataset

        Returns:
            goal seq of length validset.n_frame_stack+1
        """
        return TensorUtils.slice(x=self.validset[index]["goal"], dim=1, start=0, end=self.validset.n_frame_stack+2)

    def create_env(self):
        """
        Create environment associated with dataset.
        """
        # load env metadata from training file
        self.env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=self.config.train.data)

        if self.config.experiment.env is not None:
            self.env_meta["env_name"] = self.config.experiment.env
            print("=" * 30 + "\n" + "Replacing Env to {}\n".format(self.env_meta["env_name"]) + "=" * 30)

        self.env_name = self.env_meta["env_name"]

        # create environment for validation run
        self.env = EnvUtils.create_env_from_metadata(
                env_meta=self.env_meta,
                env_name=self.env_name, 
                render=False, 
                render_offscreen=self.config.experiment.render_video,
                use_image_obs=ObsUtils.has_modality("rgb", self.config.all_obs_keys),
                use_depth_obs=ObsUtils.has_modality("depth", self.config.all_obs_keys),
            )
        self.env = EnvUtils.wrap_env_from_config(self.env, config=self.config) # apply environment warpper, if applicable

    def run_rollout(
            self, 
            policy: RolloutPolicy, 
            demo_id,
            video_writer=None,
            video_skip=5,
            terminate_on_success=False
        ):
        """
        Args:
            demo_id (str): id of demo to rollout
        """
        super(RobomimicRolloutEnv, self).run_rollout(
            policy=policy,
            video_writer=video_writer, 
            video_skip=video_skip, 
            terminate_on_success=terminate_on_success
            )
        
        demo_len = self.validset.demo_id_to_demo_length[demo_id]
        demo_index = self.validset.demo_id_to_start_index[demo_id]
        
        policy.start_episode()

        # load initial state
        initial_state = dict(states=self.validset.hdf5_file[f"data/{demo_id}/states"][0])
        initial_state["model"] = self.validset.hdf5_file[f"data/{demo_id}"].attrs["model_file"]
        self.env.reset()
        obs = self.env.reset_to(initial_state)

        # policy inputs from initial observation
        inputs = self.inputs_from_initial_obs(obs=obs)

        results = {}
        video_count = 0  # video frame counter
        total_reward = 0.
        success = { k: False for k in self.env.is_success() } # success metrics
        try:
            for step_i in range(demo_len):
                # compute new inputs
                inputs = self.inputs_from_new_obs(x=inputs, obs=obs, index=demo_index+step_i)

                # get action from policy
                ac = policy(**inputs)

                # play action
                obs, r, done, _ = self.env.step(ac)

                # compute reward
                total_reward += r

                cur_success_metrics = self.env.is_success()
                for k in success:
                    success[k] = success[k] or cur_success_metrics[k]

                # visualization
                if video_writer is not None:
                    if video_count % video_skip == 0:
                        video_img = self.env.render(mode="rgb_array", height=256, width=256)
                        video_writer.append_data(video_img)
                    video_count += 1

                # break if done
                if done or (terminate_on_success and success["task"]):
                    break

        except self.env.rollout_exceptions as e:
            print("WARNING: got rollout exception {}".format(e))
        
        results["Return"] = total_reward
        results["Horizon"] = step_i + 1
        results["Success_Rate"] = float(success["task"])

        # log additional success metrics
        for k in success:
            if k != "task":
                results["{}_Success_Rate".format(k)] = float(success[k])

        return results
        
    def rollout_with_stats(
            self, 
            policy, 
            video_writer=None,
            epoch=None,
            video_skip=5,
            terminate_on_success=False, 
            verbose=False,
        ):   
        super(RobomimicRolloutEnv, self).rollout_with_stats(
            policy=policy,
            video_writer=video_writer, 
            epoch=epoch,
            video_skip=video_skip, 
            terminate_on_success=terminate_on_success, 
            verbose=verbose
            )
        
        # create video writer
        write_video = self.video_dir is not None
        video_path = None
        video_writer = None
        if write_video:
            video_str = "_epoch_{}.mp4".format(epoch) if epoch is not None else ".mp4" 
            video_path = os.path.join(self.video_dir, "{}".format(video_str))
            video_writer = imageio.get_writer(video_path, fps=20)
            print("video writes to " + video_path)

        rollout_logs = []
        # rollout on each demo in validset
        for i in tqdm(self.validset.num_demos):
            demo_id = self.validset.demos[i]
            rollout_timestamp = time.time()
            rollout_info = self.run_rollout(
                policy=policy, 
                demo_id=demo_id, 
                video_writer=video_writer, 
                video_skip=video_skip, 
                terminate_on_success=terminate_on_success, 
            )
            rollout_info["time"] = time.time() - rollout_timestamp
            rollout_logs.append(rollout_info)
            if verbose:
                horizon = rollout_info["Horizon"]
                num_success = rollout_info["Success_Rate"]
                print(f"demo={demo_id}, horizon={horizon}, num_success={num_success}")
                print(json.dumps(rollout_info, sort_keys=True, indent=4))

        if write_video:
            video_writer.close()

        # average metric across all episodes
        rollout_logs = dict((k, [rollout_logs[i][k] for i in range(len(rollout_logs))]) for k in rollout_logs[0])
        rollout_logs_mean = dict((k, np.mean(v)) for k, v in rollout_logs.items())
        rollout_logs_mean["Time_Episode"] = np.sum(rollout_logs["time"]) / 60. # total time taken for rollouts in minutes

        return rollout_logs_mean, video_path


        






