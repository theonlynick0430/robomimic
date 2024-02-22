from robomimic.algo import algo_factory, RolloutPolicy


class RolloutEnv:
    """
    Abstract class used to rollout policies in different environments. 
    """

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
