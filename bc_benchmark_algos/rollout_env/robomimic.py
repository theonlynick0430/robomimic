from bc_benchmark_algos.rollout_env.rollout_env import RolloutEnv
from bc_benchmark_algos.dataset.robomimic import RobomimicDataset
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import numpy as np


class RobomimicRolloutEnv(RolloutEnv):
    """
    Class used to rollout policies in in Robomimic environments. 
    """

    def __init__(self, config, validset):
        super(RobomimicRolloutEnv, self).__init__(
            config=config, 
            validset=validset, 
        )
        assert isinstance(self.validset, RobomimicDataset)
    
    def fetch_goal(self, demo_id, t):
        if not self.gc:
            return None
        demo_length = self.validset.get_demo_len(demo_id=demo_id)
        if t >= demo_length:
            # reuse last goal
            t = demo_length-1
        index = self.validset.demo_id_to_start_index[demo_id] + t
        goal = self.validset[index]["goal"]
        goal = TensorUtils.slice(x=goal, dim=0, start=0, end=self.validset.n_frame_stack+1)
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

        # create environment for validation run
        env_type = EnvUtils.get_env_type(env_meta=self.env_meta)
        self.env = EnvUtils.create_env(
            env_type=env_type,
            env_name=self.env_meta["env_name"],  
            render=False, 
            render_offscreen=self.config.experiment.render_video, 
            use_image_obs=ObsUtils.has_modality("rgb", self.config.all_obs_keys), 
            use_depth_obs=ObsUtils.has_modality("depth", self.config.all_obs_keys), 
            postprocess_visual_obs=False, # ensure shape from obs and dataset are the same
            **self.env_meta["env_kwargs"],
        )
        EnvUtils.check_env_version(self.env, self.env_meta)

    def run_rollout(
            self, 
            policy, 
            demo_id,
            video_writer=None,
            video_skip=5,
            horizon=None,
            terminate_on_success=False
        ):
        super(RobomimicRolloutEnv, self).run_rollout(
            policy=policy,
            demo_id=demo_id,
            video_writer=video_writer, 
            video_skip=video_skip, 
            horizon=horizon,
            terminate_on_success=terminate_on_success
            )
        
        demo_len = self.validset.get_demo_len(demo_id=demo_id)
        horizon = demo_len if horizon is None else horizon
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
            for step_i in range(horizon):
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


        






