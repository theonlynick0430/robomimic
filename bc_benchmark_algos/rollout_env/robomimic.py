from bc_benchmark_algos.rollout_env.rollout_env import RolloutEnv
from bc_benchmark_algos.dataset.robomimic import RobomimicDataset
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.algo import RolloutPolicy
import imageio
import numpy as np
import tqdm
import os
import time
import json
import torch


class RobomimicRolloutEnv(RolloutEnv):
    """
    Class used to rollout policies in in Robomimic environments. 
    """

    def __init__(self, config, validset, video_dir=None):
        super(RobomimicRolloutEnv, self).__init__(
            config=config, 
            validset=validset, 
            video_dir=video_dir
        )
        assert isinstance(self.validset, RobomimicDataset)
    
    def fetch_goal(self, demo_id, t):
        index = self.validset.demo_id_to_start_index[demo_id] + t
        goal = TensorUtils.slice(x=self.validset[index]["goal"], dim=0, start=0, end=self.validset.n_frame_stack+1)
        goal = TensorUtils.to_tensor(x=goal, device=self.device)
        goal = TensorUtils.to_batch(x=goal)
        return goal
        
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

    def run_rollout(
            self, 
            policy: RolloutPolicy, 
            demo_id,
            video_writer=None,
            video_skip=5,
            terminate_on_success=False
        ):
        super(RobomimicRolloutEnv, self).run_rollout(
            policy=policy,
            demo_id=demo_id,
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
        inputs = self.inputs_from_initial_obs(obs=obs, demo_id=demo_id)

        results = {}
        video_count = 0  # video frame counter
        total_reward = 0.
        success = { k: False for k in self.env.is_success() } # success metrics
        try:
            for step_i in range(demo_len):
                # compute new inputs
                inputs = self.inputs_from_new_obs(x=inputs, obs=obs, demo_id=demo_id, t=demo_index+step_i)

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
            demo_id,
            video_writer=None,
            video_skip=5,
            terminate_on_success=False, 
            verbose=False,
        ):   
        super(RobomimicRolloutEnv, self).rollout_with_stats(
            policy=policy,
            demo_id=demo_id,
            video_writer=video_writer, 
            video_skip=video_skip, 
            terminate_on_success=terminate_on_success, 
            verbose=verbose
            )

        rollout_logs = []
        rollout_timestamp = time.time()

        # create video writer
        video_path = None
        if self.write_video and video_writer is None:
            video_str = f"{demo_id}.mp4"
            video_path = os.path.join(self.video_dir, f"{video_str}")
            video_writer = imageio.get_writer(video_path, fps=20)
            print("video writes to " + video_path)
        
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

        if self.write_video:
            video_writer.close()

        # average metric across all episodes
        rollout_logs = dict((k, [rollout_logs[i][k] for i in range(len(rollout_logs))]) for k in rollout_logs[0])
        rollout_logs_mean = dict((k, np.mean(v)) for k, v in rollout_logs.items())
        rollout_logs_mean["Time_Episode"] = np.sum(rollout_logs["time"]) / 60. # total time taken for rollouts in minutes

        return rollout_logs_mean, video_path


        






