.. _observations:

Standard Observations and Actions
=================================

We have introduced `AgentInterface` in :ref:`agent` which allows us to choose from the standard observation and action types for communication
between an agent and a SMARTS environment.

============
Observations
============

Here we will introduce details of available observation types.
For `AgentType.Full`, which contains the most concrete observation details, the raw observation returned
is a Python `NamedTuple` with the following fields:

* `events` a `NamedTuple` with the following fields:
    * `collisions` - collisions the vehicle has been involved with other vehicles (if any)
    * `off_road` - `True` if the vehicle is off the road
    * `reached_goal` - `True` if the vehicle has reached its goal
    * `reached_max_episode_steps` - `True` if the vehicle has reached its max episode steps
* `ego_vehicle_state` - a `VehicleObservation` `NamedTuple` for the ego vehicle with the following fields:
    * `id` - a string identifier for this vehicle
    * `position` - Coordinate of the center of the vehicle bounding box's bottom plane. shape=(3,). dtype=np.float64.
    * `bounding_box` - `Dimensions` data class for the `length`, `width`, `height` of the vehicle
    * `heading` - vehicle heading in radians
    * `speed` - agent speed in m/s
    * `steering` - angle of front wheels in radians
    * `yaw_rate` - rotational speed in radian per second
    * `road_id` - the identifier for the road nearest to this vehicle
    * `lane_id` - a globally unique identifier of the lane under this vehicle 
    * `lane_index` - index of the lane under this vehicle, right most lane has index 0 and the index increments to the left
    * `mission` - a field describing the vehicle plotted route
    * `linear_velocity` - Vehicle velocity along body coordinate axes. A numpy array of shape=(3,) and dtype=np.float64.
    * `angular_velocity` - Angular velocity vector. A numpy array of shape=(3,) and dtype=np.float64.
    * `linear_acceleration` - Linear acceleration vector. A numpy array of shape=(3,). dtype=np.float64. Requires accelerometer sensor.
    * `angular_acceleration` - Angular acceleration vector. A numpy array of shape=(3,) and dtype=np.float64. Requires accelerometer sensor. 
    * `linear_jerk` - Linear jerk vector. A numpy array of shape=(3,) and dtype=np.float64. Requires accelerometer sensor.
    * `angular_jerk` - Angular jerk vector. A numpy array of shape=(3,) and dtype=np.float64. Requires accelerometer sensor. 
* `neighborhood_vehicle_states` - a list of `VehicleObservation` `NamedTuple`s, each with the following fields:
    * `position`, `bounding_box`, `heading`, `speed`, `lane_id`, `lane_index` - the same as with `ego_vehicle_state`
* `GridMapMetadata` - Metadata for the observation maps with the following information,
    * `created_at` - time at which the map was loaded
    * `resolution` - map resolution in world-space-distance/cell
    * `width` - map width in # of cells
    * `height` - map height in # of cells
    * `camera_pos` - camera position when project onto the map
    * `camera_heading_in_degrees` - camera rotation angle along z-axis when project onto the map
* `top_down_rgb` - contains an observation image with its metadata
    * `metadata` - `GridMapMetadata`
    * `data` - a RGB image (default 256x256) with the ego vehicle at the center

.. image:: ../_static/rgb.png

* `occupancy_grid_map` - contains an observation image with its metadata
    * `metadata` - `GridMapMetadata`
    * `data` - An `OGM <https://en.wikipedia.org/wiki/Occupancy_grid_mapping>`_ (default 256x256) around the ego vehicle
* `drivable_area_grid_map` - contains an observation image with its metadata
    * `metadata` - `GridMapMetadata`
    * `data` - A grid map (default 256x256) that shows the static drivable area around the ego vehicle
* `waypoint_paths` - A list of waypoints in front of the ego vehicle showing the potential routes ahead. Each item is a `Waypoint` instance with the following fields:
    * `id` - an integer identifier for this waypoint
    * `pos` - a numpy array (x, y) center point along the lane
    * `heading` - heading angle of lane at this point (radians)
    * `lane_width` - width of lane at this point (meters)
    * `speed_limit` - lane speed in m/s
    * `lane_id` - a globally unique identifier of lane under waypoint
    * `right_of_way` - `True` if this waypoint has right of way, `False` otherwise
    * `lane_index` - index of the lane under this waypoint, right most lane has index 0 and the index increments to the left

See implementation in :class:`smarts.core.sensors`


Then, you can choose the observations needed through :class:`smarts.core.agent_interface.AgentInterface` and process these raw observations through :class:`smarts.core.observation_adapter`.
Note: Some observations like `occupancy_grid_map`, `drivable_area_grid_map` and `top_down_rgb` requires the use of Panda3D package to render agent camera observations during simulations. So you need to install the required dependencies first using the command `pip install -e .[camera-obs]`

=======
Rewards
=======
The reward from smarts environments is given by a calculation within smarts; `env_reward` from smarts environments directly uses the reward from smarts. The given reward is 0 or `reward < -0.5` or `reward > 0.5` relating to distance traveled in meters on the step that a vehicle has gone at least 0.5 meters since the last given non-zero reward.

=======
Actions
=======

* `ActionSpaceType.Continuous`: continuous action space with throttle, brake, absolute steering angle. It is a tuple of `throttle` [0, 1], `brake` [0, 1], and `steering` [-1, 1].
* `ActionSpaceType.ActuatorDynamic`: continuous action space with throttle, brake, steering rate. Steering rate means the amount of steering angle change *per second* (either positive or negative) to be applied to the current steering angle. It is also a tuple of `throttle` [0, 1], `brake` [0, 1], and `steering_rate`, where steering rate is in number of radians per second.
* `ActionSpaceType.Lane`: discrete lane action space of *strings* including "keep_lane",  "slow_down", "change_lane_left", "change_lane_right" as of version 0.3.2b, but a newer version will soon be released. In this newer version, the action space will no longer consist of strings, but will be a tuple of an integer for `lane_change` and a float for `target_speed`.
