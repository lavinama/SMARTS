# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
import math

import numpy as np
import pytest

from smarts.core.coordinates import Point
from smarts.core.opendrive_road_network import OpenDriveRoadNetwork
from smarts.core.scenario import Scenario
from smarts.core.sumo_road_network import SumoRoadNetwork


@pytest.fixture
def sumo_scenario():
    return Scenario(scenario_root="scenarios/intersections/4lane")


@pytest.fixture
def opendrive_scenario_4lane():
    return Scenario(scenario_root="scenarios/od_4lane")


@pytest.fixture
def opendrive_scenario_merge():
    return Scenario(scenario_root="scenarios/od_merge")


def test_sumo_map(sumo_scenario):
    road_map = sumo_scenario.road_map
    assert isinstance(road_map, SumoRoadNetwork)

    point = (125.20, 139.0, 0)
    lane = road_map.nearest_lane(point)
    assert lane.lane_id == "edge-north-NS_0"
    assert lane.road.road_id == "edge-north-NS"
    assert lane.index == 0
    assert lane.road.contains_point(point)
    assert lane.is_drivable
    assert len(lane.shape().exterior.coords) >= 2
    assert lane.length == 55.6

    right_lane, direction = lane.lane_to_right
    assert not right_lane

    left_lane, direction = lane.lane_to_left
    assert left_lane
    assert direction
    assert left_lane.lane_id == "edge-north-NS_1"
    assert left_lane.index == 1

    lefter_lane, direction = left_lane.lane_to_left
    assert not lefter_lane

    on_roads = lane.road.oncoming_roads_at_point(point)
    assert on_roads
    assert len(on_roads) == 1
    assert on_roads[0].road_id == "edge-north-SN"

    reflinept = lane.to_lane_coord(point)
    assert reflinept.s == 1.0
    assert reflinept.t == 0.0

    offset = reflinept.s
    assert lane.width_at_offset(offset) == 3.2
    assert lane.curvature_radius_at_offset(offset) == math.inf

    on_lanes = lane.oncoming_lanes_at_offset(offset)
    assert not on_lanes
    on_lanes = left_lane.oncoming_lanes_at_offset(offset)
    assert len(on_lanes) == 1
    assert on_lanes[0].lane_id == "edge-north-SN_1"

    in_lanes = lane.incoming_lanes
    assert not in_lanes

    out_lanes = lane.outgoing_lanes
    assert out_lanes
    assert len(out_lanes) == 2
    assert out_lanes[0].lane_id == ":junction-intersection_0_0"
    assert out_lanes[1].lane_id == ":junction-intersection_1_0"

    foes = out_lanes[0].foes
    assert foes
    assert len(foes) == 3
    foe_set = set(f.lane_id for f in foes)
    assert "edge-east-EW_0" in foe_set  # entering from east
    assert "edge-north-NS_0" in foe_set  # entering from north
    assert ":junction-intersection_5_0" in foe_set  # crossing from east-to-west

    # Test the lane vector for a refline point outside lane
    lane_heading_at_offset = lane.vector_at_offset(55.7)
    assert np.array_equal(lane_heading_at_offset, np.array([0.0, -1.0, 0.0]))

    r1 = road_map.road_by_id("edge-north-NS")
    assert r1
    assert r1.is_drivable
    assert len(r1.shape()) >= 2
    r2 = road_map.road_by_id("edge-east-WE")
    assert r2
    assert r2.is_drivable
    assert len(r2.shape()) >= 2

    routes = road_map.generate_routes(r1, r2)
    assert routes
    assert len(routes[0].roads) == 4

    route = routes[0]
    db = route.distance_between(point, (198, 65.20, 0))
    assert db == 134.01

    cands = route.project_along(point, 134.01)
    for r2lane in r2.lanes:
        assert (r2lane, 53.6) in cands

    cands = left_lane.project_along(offset, 134.01)
    assert len(cands) == 6
    for r2lane in r2.lanes:
        if r2lane.index == 1:
            assert any(
                r2lane == cand[0] and math.isclose(cand[1], 53.6) for cand in cands
            )


def test_opendrive_map_4lane(opendrive_scenario_4lane):
    road_map = opendrive_scenario_4lane.road_map
    assert isinstance(road_map, OpenDriveRoadNetwork)

    # Expected properties for all roads and lanes
    for road_id, road in road_map._roads.items():
        assert type(road_id) == str
        assert road.is_junction is not None
        assert road.length is not None
        assert road.length >= 0
        assert road.parallel_roads == []
        for lane in road.lanes:
            assert lane.in_junction is not None
            assert lane.length is not None
            assert lane.length >= 0
            assert lane.speed_limit >= 0

    # Road tests
    r_0_R = road_map.road_by_id("57_0_R")
    assert r_0_R
    assert not r_0_R.is_junction
    assert r_0_R.length == 115.6
    assert len(r_0_R.lanes) == 2
    assert r_0_R.lane_at_index(-1) is None
    assert r_0_R.lane_at_index(1).road.road_id == "57_0_R"
    assert len(r_0_R.shape().exterior.coords) == 5
    assert set(r.road_id for r in r_0_R.incoming_roads) == set()
    assert set(r.road_id for r in r_0_R.outgoing_roads) == {
        "69_0_R",
        "68_0_R",
        "67_0_R",
    }

    r2_0_R = road_map.road_by_id("53_0_R")
    assert r2_0_R
    assert not r2_0_R.is_junction
    assert r2_0_R.length == 55.6
    assert len(r2_0_R.lanes) == 2
    assert r2_0_R.lane_at_index(0).road.road_id == "53_0_R"
    assert len(r2_0_R.shape().exterior.coords) == 5
    assert set(r.road_id for r in r2_0_R.incoming_roads) == {
        "61_0_R",
        "65_0_R",
        "69_0_R",
    }
    assert set(r.road_id for r in r2_0_R.outgoing_roads) == set()

    r3_0_R = road_map.road_by_id("66_0_R")
    assert r3_0_R
    assert r3_0_R.is_junction
    assert r3_0_R.length == 23.35459442
    assert len(r3_0_R.lanes) == 1
    assert r3_0_R.lane_at_index(-1) is None
    assert r3_0_R.lane_at_index(0).road.road_id == "66_0_R"
    assert set(r.road_id for r in r3_0_R.incoming_roads) == {"55_0_R"}
    assert set(r.road_id for r in r3_0_R.outgoing_roads) == {"56_0_R"}

    r4_0_R = road_map.road_by_id("59_0_R")
    assert r4_0_R
    assert r4_0_R.is_junction
    assert r4_0_R.length == 28.8
    assert len(r4_0_R.lanes) == 2
    assert r4_0_R.lane_at_index(0).road.road_id == "59_0_R"
    assert set(r.road_id for r in r4_0_R.incoming_roads) == {"52_0_R"}
    assert set(r.road_id for r in r4_0_R.outgoing_roads) == {"54_0_R"}

    # Lane tests
    l1 = road_map.lane_by_id("52_0_R_-1")
    assert l1
    assert l1.road.road_id == "52_0_R"
    # leftmost lane
    assert l1.index == 1
    assert len(l1.lanes_in_same_direction) == 1
    assert l1.length == 55.6
    assert l1.is_drivable
    assert len(l1.shape().exterior.coords) == 5
    assert l1.speed_limit == 16.67
    assert set(l.lane_id for l in l1.incoming_lanes) == set()
    assert set(l.lane_id for l in l1.outgoing_lanes) == {
        "58_0_R_-1",
        "59_0_R_-1",
        "60_0_R_-1",
    }

    right_lane, direction = l1.lane_to_right
    assert right_lane
    assert direction
    assert right_lane.lane_id == "52_0_R_-2"
    assert right_lane.index == 0

    further_right_lane, direction = right_lane.lane_to_right
    assert not further_right_lane
    assert direction

    left_lane, direction = l1.lane_to_left
    assert left_lane
    assert not direction
    assert left_lane.lane_id == "53_0_R_-1"
    assert left_lane.index == 1

    # point on lane
    point = (148.0, -28.0, 0)
    refline_pt = l1.to_lane_coord(point)
    assert round(refline_pt.s, 2) == 38.0
    assert round(refline_pt.t, 2) == -0.4

    offset = refline_pt.s
    assert l1.width_at_offset(offset) == 3.20
    assert l1.curvature_radius_at_offset(offset) == math.inf
    assert l1.contains_point(point)
    assert l1.road.contains_point(point)
    central_point = l1.center_at_point(point)
    assert (round(central_point.x, 2), round(central_point.y, 2)) == (148.4, -28.0)

    # oncoming lanes at this point
    on_lanes = l1.oncoming_lanes_at_offset(offset)
    assert on_lanes
    assert len(on_lanes) == 1
    assert on_lanes[0].lane_id == "53_0_R_-1"

    # lane edges on point
    left_edge, right_edge = l1.edges_at_point(point)
    assert (round(left_edge.x, 2), round(left_edge.y, 2)) == (150.0, -28.0)
    assert (round(right_edge.x, 2), round(right_edge.y, 2)) == (146.8, -28.0)

    # road edges on point
    road_left_edge, road_right_edge = l1.road.edges_at_point(point)
    assert (round(road_left_edge.x, 2), round(road_left_edge.y, 2)) == (150.0, -28.0)
    assert (round(road_right_edge.x, 2), round(road_right_edge.y, 2)) == (143.6, -28.0)

    # check for locations (lane, offset tuples) within distance at this offset
    candidates = l1.project_along(offset, 70)
    assert (len(candidates)) == 6

    # point not on lane but on road
    point = (144.0, -28.0, 0)
    refline_pt = l1.to_lane_coord(point)
    assert round(refline_pt.s, 2) == 38.0
    assert round(refline_pt.t, 2) == -4.4

    offset = refline_pt.s
    assert l1.width_at_offset(offset) == 3.20
    assert l1.curvature_radius_at_offset(offset) == math.inf
    assert not l1.contains_point(point)
    assert l1.road.contains_point(point)

    l2 = road_map.lane_by_id("63_0_R_-1")
    assert l2
    assert l2.road.road_id == "63_0_R"
    assert l2.index == 0
    assert l2.is_drivable
    assert [l.lane_id for l in l2.incoming_lanes] == ["50_0_R_-1"]
    assert [l.lane_id for l in l2.outgoing_lanes] == ["54_0_R_-1"]

    l3 = road_map.lane_by_id("66_0_R_-1")
    assert l3
    assert l3.road.road_id == "66_0_R"
    assert l3.index == 0
    assert l3.is_drivable

    foes = l3.foes
    assert set(f.lane_id for f in foes) == {"58_0_R_-1", "62_0_R_-1"}

    # road edges on point for a road with one lane
    point = (148.52, -52.22, 0)
    r5 = road_map.road_by_id("60_0_R")
    road_left_edge, road_right_edge = r5.edges_at_point(point)
    assert (round(road_left_edge.x, 2), round(road_left_edge.y, 2)) == (150.79, -51.56)
    assert (round(road_right_edge.x, 2), round(road_right_edge.y, 2)) == (
        147.72,
        -52.45,
    )

    # nearest lane for a point outside road
    point = (164.0, -68.0, 0)
    l4 = road_map.nearest_lane(point)
    assert l4.lane_id == "64_0_R_-2"
    assert l4.road.road_id == "64_0_R"
    assert l4.index == 0
    assert l4.speed_limit == 16.67
    assert not l4.road.contains_point(point)
    assert l4.is_drivable

    # nearest lane for a point inside road
    point = (151.0, -60.0, 0)
    l5 = road_map.nearest_lane(point)
    assert l5.lane_id == "65_0_R_-1"
    assert l5.road.road_id == "65_0_R"
    assert l5.index == 1
    assert l5.road.contains_point(point)
    assert l5.is_drivable

    # Lanepoints
    lanepoints = road_map._lanepoints

    point = (148.0, -17.0, 0)
    l1_lane_point = lanepoints.closest_lanepoint_on_lane_to_point(point, l1.lane_id)
    assert (
        round(l1_lane_point.pose.position[0], 2),
        round(l1_lane_point.pose.position[1], 2),
    ) == (148.4, -17.0)

    point = (148.00, -47.00)
    r5_linked_lane_point = lanepoints.closest_linked_lanepoint_on_road(
        point, r5.road_id
    )
    assert r5_linked_lane_point.lp.lane.lane_id == "60_0_R_-1"
    assert (
        round(r5_linked_lane_point.lp.pose.position[0], 2),
        round(r5_linked_lane_point.lp.pose.position[1], 2),
    ) == (148.43, -46.88)

    r5_lp_path = lanepoints.paths_starting_at_lanepoint(r5_linked_lane_point, 5, ())
    assert len(r5_lp_path) == 1
    assert [llp.lp.lane.lane_id for llp in r5_lp_path[0]].count("60_0_R_-1") == 6

    # route generation
    r_52_0_R = road_map.road_by_id("52_0_R")
    r_58_0_R = road_map.road_by_id("58_0_R")
    r_56_0_R = road_map.road_by_id("56_0_R")

    route_52_to_56 = road_map.generate_routes(r_52_0_R, r_56_0_R)
    assert [r.road_id for r in route_52_to_56[0].roads] == [
        "52_0_R",
        "58_0_R",
        "56_0_R",
    ]
    assert (
        route_52_to_56[0].road_length
        == r_52_0_R.length + r_58_0_R.length + r_56_0_R.length
    )

    # waypoints generation along route
    lp_52_0_R = road_map._lanepoints._lanepoints_by_lane_id["52_0_R_-1"]
    lp_pose = lp_52_0_R[-1].lp.pose
    waypoints_for_route = road_map.waypoint_paths(lp_pose, 170, route=route_52_to_56[0])
    assert len(waypoints_for_route) == 8
    assert len(waypoints_for_route[1]) == 165
    lane_ids_under_wps = []
    for wp in waypoints_for_route[0]:
        if wp.lane_id not in lane_ids_under_wps:
            lane_ids_under_wps.append(wp.lane_id)
    assert lane_ids_under_wps == ["52_0_R_-1", "58_0_R_-1", "56_0_R_-1"]

    # distance between points along route
    start_point = Point(x=148.0, y=-28.0, z=0.0)
    end_point = Point(x=116.0, y=-58.0, z=0.0)
    assert round(route_52_to_56[0].distance_between(start_point, end_point), 2) == 60.55

    # project along route
    candidates = route_52_to_56[0].project_along(start_point, 100)
    assert len(candidates) == 2

    r_50_0_R = road_map.road_by_id("50_0_R")
    r_63_0_R = road_map.road_by_id("63_0_R")
    r_54_0_R = road_map.road_by_id("54_0_R")
    route_50_to_54 = road_map.generate_routes(r_50_0_R, r_54_0_R)
    assert [r.road_id for r in route_50_to_54[0].roads] == [
        "50_0_R",
        "63_0_R",
        "54_0_R",
    ]
    assert (
        route_50_to_54[0].road_length
        == r_50_0_R.length + r_63_0_R.length + r_54_0_R.length
    )

    # waypoints generation along route
    lp_50_0_R = road_map._lanepoints._lanepoints_by_lane_id["50_0_R_-1"]
    lp_pose = lp_50_0_R[-1].lp.pose
    waypoints_for_route = road_map.waypoint_paths(lp_pose, 110, route=route_50_to_54[0])
    assert len(waypoints_for_route) == 2
    assert len(waypoints_for_route[1]) == 105
    lane_ids_under_wps = []
    for wp in waypoints_for_route[0]:
        if wp.lane_id not in lane_ids_under_wps:
            lane_ids_under_wps.append(wp.lane_id)
    assert lane_ids_under_wps == ["50_0_R_-1", "63_0_R_-1", "54_0_R_-1"]

    # distance between points along route
    start_point = Point(x=176.0, y=-58.0, z=0.0)
    end_point = Point(x=148.0, y=-121.0, z=0.0)
    assert round(route_50_to_54[0].distance_between(start_point, end_point), 2) == 81.55

    # project along route
    candidates = route_50_to_54[0].project_along(start_point, 100)
    assert len(candidates) == 0

    # Invalid route generation
    invalid_route = road_map.generate_routes(
        road_map.road_by_id("50_0_R"), road_map.road_by_id("55_0_R")
    )
    assert [r.road_id for r in invalid_route[0].roads] == []


def test_opendrive_map_merge(opendrive_scenario_merge):
    road_map = opendrive_scenario_merge.road_map
    assert isinstance(road_map, OpenDriveRoadNetwork)
    assert road_map.bounding_box.max_pt == Point(x=100.0, y=9.750000000000002, z=0)
    assert road_map.bounding_box.min_pt == Point(x=0.0, y=-6.5, z=0)

    # Expected properties for all roads and lanes
    for road_id, road in road_map._roads.items():
        assert type(road_id) == str
        assert road.is_junction is not None
        assert road.length is not None
        assert road.length >= 0
        assert road.parallel_roads == []
        for lane in road.lanes:
            assert lane.in_junction is not None
            assert lane.length is not None
            assert lane.length >= 0
            assert lane.speed_limit == 16.67

    # Nonexistent road/lane tests
    assert road_map.road_by_id("") is None
    assert road_map.lane_by_id("") is None

    # Surface tests
    surface = road_map.surface_by_id("1_1_R")
    assert surface.surface_id == "1_1_R"

    # Road tests
    r_1_0_R = road_map.road_by_id("1_0_R")
    assert r_1_0_R
    assert len(r_1_0_R.lanes) == 2
    assert not r_1_0_R.is_junction
    assert set(r.road_id for r in r_1_0_R.incoming_roads) == set()
    assert set(r.road_id for r in r_1_0_R.outgoing_roads) == {"1_1_R"}

    r_1_0_L = road_map.road_by_id("1_0_L")
    assert r_1_0_L
    assert len(r_1_0_L.lanes) == 3
    assert not r_1_0_L.is_junction
    assert set(r.road_id for r in r_1_0_L.incoming_roads) == {"1_1_L"}
    assert set(r.road_id for r in r_1_0_L.outgoing_roads) == set()

    r_1_1_R = road_map.road_by_id("1_1_R")
    assert r_1_1_R
    assert len(r_1_1_R.lanes) == 3
    assert not r_1_1_R.is_junction
    assert set(r.road_id for r in r_1_1_R.incoming_roads) == {"1_0_R"}
    assert set(r.road_id for r in r_1_1_R.outgoing_roads) == {"1_2_R"}
    assert set(s.surface_id for s in r_1_1_R.entry_surfaces) == {"1_0_R"}
    assert set(s.surface_id for s in r_1_1_R.exit_surfaces) == {"1_2_R"}

    r_1_1_L = road_map.road_by_id("1_1_L")
    assert r_1_1_L
    assert len(r_1_1_L.lanes) == 3
    assert not r_1_1_L.is_junction
    assert set(r.road_id for r in r_1_1_L.incoming_roads) == {"1_2_L"}
    assert set(r.road_id for r in r_1_1_L.outgoing_roads) == {"1_0_L"}
    assert set(s.surface_id for s in r_1_1_L.entry_surfaces) == {"1_2_L"}
    assert set(s.surface_id for s in r_1_1_L.exit_surfaces) == {"1_0_L"}

    r_1_2_R = road_map.road_by_id("1_2_R")
    assert r_1_2_R
    assert len(r_1_2_R.lanes) == 3
    assert not r_1_2_R.is_junction
    assert set(r.road_id for r in r_1_2_R.incoming_roads) == {"1_1_R"}
    assert set(r.road_id for r in r_1_2_R.outgoing_roads) == set()

    r_1_2_L = road_map.road_by_id("1_2_L")
    assert r_1_2_L
    assert len(r_1_2_L.lanes) == 2
    assert not r_1_2_L.is_junction
    assert set(r.road_id for r in r_1_2_L.incoming_roads) == set()
    assert set(r.road_id for r in r_1_2_L.outgoing_roads) == {"1_1_L"}

    # Lane tests
    l0 = road_map.lane_by_id("1_1_L_1")
    assert l0
    assert l0.road.road_id == "1_1_L"
    assert l0.index == 2
    assert l0.is_drivable

    assert set(lane.lane_id for lane in l0.incoming_lanes) == set()
    assert set(lane.lane_id for lane in l0.outgoing_lanes) == {"1_0_L_1"}
    assert set(lane.lane_id for lane in l0.entry_surfaces) == set()
    assert set(lane.lane_id for lane in l0.exit_surfaces) == {"1_0_L_1"}

    right_lane, direction = l0.lane_to_right
    assert right_lane
    assert direction
    assert right_lane.lane_id == "1_1_L_2"
    assert right_lane.index == 1

    left_lane, direction = l0.lane_to_left
    assert left_lane
    assert not direction
    assert left_lane.lane_id == "1_1_R_-1"
    assert left_lane.index == 2

    further_right_lane, direction = right_lane.lane_to_right
    assert further_right_lane
    assert direction
    assert further_right_lane.lane_id == "1_1_L_3"
    assert further_right_lane.is_drivable
    assert set(lane.lane_id for lane in further_right_lane.outgoing_lanes) == {
        "1_0_L_3"
    }
    assert further_right_lane.index == 0

    l1 = road_map.lane_by_id("1_1_R_-1")
    assert l1
    assert l1.road.road_id == "1_1_R"
    assert l1.index == 2
    assert l1.is_drivable

    left_lane, direction = l1.lane_to_left
    assert left_lane
    assert not direction
    assert left_lane.lane_id == "1_1_L_1"
    assert left_lane.index == 2

    l2 = road_map.lane_by_id("1_1_R_-2")
    assert [l.lane_id for l in l2.incoming_lanes] == []

    # Test for lane vector when lane offset is outside lane
    l0_vector = l0.vector_at_offset(50.01)
    l0_vector = l0_vector.tolist()
    assert l0_vector == [-0.9999973500028005, -0.0020437969740241257, 0.0]

    # point on lane
    point = (31.0, 2.0, 0)
    refline_pt = l0.to_lane_coord(point)
    assert round(refline_pt.s, 2) == 44.02
    assert round(refline_pt.t, 2) == -0.31

    offset = refline_pt.s
    assert round(l0.width_at_offset(offset), 2) == 3.12
    assert round(l0.curvature_radius_at_offset(offset), 2) == -291.53
    assert l0.contains_point(point)
    assert l0.road.contains_point(point)

    # lane edges on point
    left_edge, right_edge = l0.edges_at_point(point)
    assert (round(left_edge.x, 2), round(left_edge.y, 2)) == (31.08, 0.13)
    assert (round(right_edge.x, 2), round(right_edge.y, 2)) == (31.0, 3.25)

    # road edges on point
    road_left_edge, road_right_edge = r_1_1_R.edges_at_point(point)
    assert (round(road_left_edge.x, 2), round(road_left_edge.y, 2)) == (31.08, 0.13)
    assert (round(road_right_edge.x, 2), round(road_right_edge.y, 2)) == (31.0, -6.5)

    # point not on lane but on road
    point = (31.0, 4.5, 0)
    refline_pt = l0.to_lane_coord(point)
    assert round(refline_pt.s, 2) == 43.97
    assert round(refline_pt.t, 2) == -2.81

    offset = refline_pt.s
    assert round(l0.width_at_offset(offset), 2) == 3.12
    assert round(l0.curvature_radius_at_offset(offset), 2) == -292.24
    assert not l0.contains_point(point)
    assert l0.road.contains_point(point)

    # check for locations (lane, offset tuples) within distance at this offset
    candidates = l0.project_along(offset, 20)
    assert (len(candidates)) == 3

    # nearest lanes for a point in lane
    point = (60.0, -2.38, 0)
    l4 = road_map.nearest_lane(point)
    assert l4.lane_id == "1_1_R_-2"
    assert l4.road.road_id == "1_1_R"
    assert l4.index == 1
    assert l4.road.contains_point(point)
    assert l4.is_drivable

    # get the road for point containing it
    point = (80.0, 1.3, 0)
    r4 = road_map.road_with_point(point)
    assert r4.road_id == "1_2_R"

    # route generation
    start = road_map.road_by_id("1_0_R")
    end = road_map.road_by_id("1_2_R")
    route = road_map.generate_routes(start, end)
    assert [r.road_id for r in route[0].roads] == ["1_0_R", "1_1_R", "1_2_R"]

    # waypoints generation along route
    lp_1_0_R = road_map._lanepoints._lanepoints_by_lane_id["1_0_R_-1"]
    lp_pose = lp_1_0_R[0].lp.pose
    waypoints_for_route = road_map.waypoint_paths(lp_pose, 60, route=route[0])
    assert len(waypoints_for_route) == 6
    assert len(waypoints_for_route[0]) == 61
    lane_ids_under_wps = frozenset(
        [
            frozenset([wp.lane_id for wp in waypoints_for_route[i]])
            for i in range(len(waypoints_for_route))
        ]
    )
    assert {"1_0_R_-1", "1_1_R_-1"} in lane_ids_under_wps
    assert {"1_0_R_-2", "1_1_R_-3"} in lane_ids_under_wps

    # distance between points along route
    start_point = Point(x=17.56, y=-1.67, z=0.0)
    end_point = Point(x=89.96, y=2.15, z=0.0)
    assert round(route[0].distance_between(start_point, end_point), 2) == 72.4
    # project along route
    candidates = route[0].project_along(start_point, 70)
    assert len(candidates) == 3

    # Lanepoints
    lanepoints = road_map._lanepoints

    point = (48.39, 0.4, 0)
    l1_lane_point = lanepoints.closest_lanepoint_on_lane_to_point(point, "1_1_R_-1")
    assert (
        round(l1_lane_point.pose.position[0], 2),
        round(l1_lane_point.pose.position[1], 2),
    ) == (48.5, -0.15)

    point = (20.0, 1.3, 0)
    r0_linked_lane_point = lanepoints.closest_linked_lanepoint_on_road(point, "1_0_L")
    assert r0_linked_lane_point.lp.lane.lane_id == "1_0_L_1"
    assert (
        round(r0_linked_lane_point.lp.pose.position[0], 2),
        round(r0_linked_lane_point.lp.pose.position[1], 2),
    ) == (20.0, 1.62)

    r0_lp_path = lanepoints.paths_starting_at_lanepoint(r0_linked_lane_point, 5, ())
    assert len(r0_lp_path) == 1
    assert [llp.lp.lane.lane_id for llp in r0_lp_path[0]].count("1_0_L_1") == 6
