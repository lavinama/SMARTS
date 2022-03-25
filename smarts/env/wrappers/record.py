# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import shutil
import gym
import os
from typing import Any, Dict, Tuple

import numpy as np

from smarts.core.sensors import Observation
from smarts.core.utils.file import smarts_log_dir
from smarts.env.wrappers.utils.rendering import vis_sim_obs

Action = Any
Operation = Any

default_log_dir = smarts_log_dir()


class AgentCameraRGBRender(gym.Wrapper):
    """Wraps the environment with `rgb_array` render mode capabilities."""

    def __init__(self, env: gym.Env, max_agents: int = 1, enabled=True):
        super().__init__(env)
        self.env.metadata["render.modes"] = set(self.env.metadata["render.modes"]) | {
            "rgb_array"
        }
        super(AgentCameraRGBRender, self).__init__(env)
        self._max_agents = max_agents
        self._current_agents = {}
        self._enabled = enabled
        self._image_frame = []
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def step(self, action: Action) -> Tuple[Operation, float, bool, Dict[str, Any]]:
        """Record a step."""

        obs, rewards, dones, infos = super().step(action)

        single_env_obs = obs
        if self.is_vector_env:
            # For now only render one environment
            single_env_obs = obs[0]

        self._record_for_render(single_env_obs)

        return (obs, rewards, dones, infos)

    def render(self, mode="rgb_array", **kwargs):
        super().render(mode, **kwargs)

        if mode == "rgb_array" and len(self._image_frame) > 0:
            return self._image_frame

    def reset(self) -> Any:
        """Record the reset of the environment."""

        try:
            os.mkdir(self._recording_dir)
        except:
            pass
        obs = super().reset()
        self._record_for_render(obs)
        self._current_agents = set()
        return obs

    def _record_for_render(self, obs) -> Any:
        if not self._enabled:
            return
        if isinstance(obs, Observation):
            obs = {"default_agent": obs}
        values = vis_sim_obs(obs).values()
        images = np.stack(values, axis=0)[0]

        if len(images.shape) == 0:
            return
        largest_image = max(images, key=lambda im: np.product(im.shape))

        image = np.array([np.resize(im, largest_image.shape) for im in images])
        if len(image.shape) > 2:
            self._image_frame = np.reshape(
                image, (image.shape[0] * image.shape[2], *image.shape[2:])
            )