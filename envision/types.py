# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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
from enum import Enum
from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

from smarts.core.events import Events


class TrafficActorType(str, Enum):
    """Traffic actor type information to help distinguish actors from each other."""

    SocialVehicle = "social_vehicle"
    SocialAgent = "social_agent"
    Agent = "agent"


class VehicleType(str, Enum):
    """Vehicle classification type information."""

    Bus = "bus"
    Coach = "coach"
    Truck = "truck"
    Trailer = "trailer"
    Car = "car"


class TrafficActorState(NamedTuple):
    """Individual traffic actor state and meta information."""

    actor_type: TrafficActorType
    vehicle_type: Union[VehicleType, str]  # TODO: Restrict to VehicleType only
    position: Tuple[float, float, float]
    heading: float
    speed: float
    name: str = ""
    actor_id: Optional[str] = None
    events: Events = None
    waypoint_paths: Sequence = []
    driven_path: Sequence = []
    point_cloud: Sequence = []
    mission_route_geometry: Sequence[Sequence[Tuple[float, float]]] = None


class State(NamedTuple):
    """A full representation of a single frame of an envision simulation step."""

    traffic: Dict[str, TrafficActorState]
    scenario_id: str
    scenario_name: str
    # sequence of x, y coordinates
    bubbles: Sequence[Sequence[Tuple[float, float]]]
    scene_colors: Dict[str, Tuple[float, float, float, float]]
    scores: Dict[str, float]
    ego_agent_ids: list
    position: Dict[str, Tuple[float, float]]
    speed: Dict[str, float]
    heading: Dict[str, float]
    lane_ids: Dict[str, str]
    frame_time: float


def format_actor_id(actor_id: str, vehicle_id: str, is_multi: bool):
    """A conversion utility to ensure that an actor id conforms to envision's actor id standard.
    Args:
        actor_id: The base id of the actor.
        vehicle_id: The vehicle id of the vehicle the actor associates with.
        is_multi: If an actor associates with multiple vehicles.

    Returns:
        An envision compliant actor id.
    """
    if is_multi:
        return f"{actor_id}-{{{vehicle_id[:4]}}}"
    return actor_id
