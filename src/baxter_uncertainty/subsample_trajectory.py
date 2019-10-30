#!/usr/bin/python

import xtf.xtf as XTF
import math
from copy import deepcopy
import os
import string
import random
import rospy
from baxter_uncertainty.srv import *

def subsample_trajectories(trajectory_directory):
    # Get the list of trajectory files
    print("Starting to subsample trajectories")
    file_list = os.listdir(trajectory_directory)
    xtf_files = []
    for filename in file_list:
        if "executed.xtf" in filename:
            xtf_files.append(filename)
    # For each file, subsample
    for filename in xtf_files:
        original_filepath = trajectory_directory + filename
        new_filepath = string.replace(original_filepath, "executed.xtf", "subsampled.xtf")
        subsample_trajectory(original_filepath, new_filepath)
    print("Completed subsampling")

def subsample_trajectory(original_filepath, new_filepath):
    #raw_input("Start subsampling file: " + original_filepath + "...")
    parser = XTF.XTFParser()
    feature_client = rospy.ServiceProxy("compute_baxter_cost_features", ComputeFeatures, persistent=True)
    original_trajectory = parser.ParseTraj(original_filepath)
    new_traj = deepcopy(original_trajectory)
    new_traj.trajectory = []
    current_sequence = 0
    # Update the first point
    first_point = original_trajectory.trajectory[0]
    first_point.extras["reference_position"] = deepcopy(first_point.position_desired)
    # Add the first point
    new_traj.trajectory.append(first_point)
    # Go through and subsample
    for index in xrange(1, len(original_trajectory.trajectory)):
        start_point = original_trajectory.trajectory[index - 1]
        end_point = original_trajectory.trajectory[index]
        # Copy the original position (w/o added noise)
        end_point.extras["reference_position"] = deepcopy(end_point.position_desired)
        # Add noise to the end point
        sampled_config = sample_resultant_state(start_point.position_desired, end_point.position_desired)
        end_point.position_desired = sampled_config
        # Subsample between them
        subsampled_sequence = subsample_between_points(feature_client, start_point, end_point, current_sequence)
        new_traj.trajectory += subsampled_sequence
        current_sequence += len(subsampled_sequence)
    raw_input("Run through reference positions and compute reference cost...")
    for state in new_traj.trajectory:
        req = ComputeFeaturesRequest()
        req.ArmOption = ComputeFeaturesRequest.LEFT_ARM_ONLY
        req.FeatureOption = ComputeFeaturesRequest.COST_ONLY
        req.LeftArmConfiguration = state.extras["reference_position"]
        req.GradientMultiplier = 0.1
        res = feature_client.call(req)
        state.extras["reference_cost"] = res.LeftArmCost
    raw_input("Run through sampled positions and compute executed cost...")
    for state in new_traj.trajectory:
        req = ComputeFeaturesRequest()
        req.ArmOption = ComputeFeaturesRequest.LEFT_ARM_ONLY
        req.FeatureOption = ComputeFeaturesRequest.COST_ONLY
        req.LeftArmConfiguration = state.position_desired
        req.GradientMultiplier = 0.1
        res = feature_client.call(req)
        state.extras["state_cost"] = res.LeftArmCost
    #raw_input("Finish processing file...")
    parser.ExportTraj(new_traj, new_filepath)

def sample_resultant_state(current_configuration, target_configuration):
    # For each joint, sample a new joint value given the current state and control input (target-current)
    # Determine the control input
    control_input = [0.0 for idx in xrange(len(current_configuration))]
    for index in range(len(current_configuration)):
        control_input[index] = target_configuration[index] - current_configuration[index]
    # Sample a new configuration based on the control input
    sampled_configuration = [0.0 for idx in xrange(len(current_configuration))]
    for index in range(len(current_configuration)):
        sampled_configuration[index] = sample_joint(current_configuration[index], control_input[index])
    return sampled_configuration

def sample_joint(current_value, control_input):
    mean = current_value + control_input
    variance = (0.001 * control_input) + 0.001
    sigma = math.sqrt(abs(variance))
    return random.gauss(mean, sigma)

def subsample_between_points(feature_client, start_point, end_point, current_sequence, increment=0.01):
    distance = euclidean_distance(start_point.position_desired, end_point.position_desired)
    increments = int(round(distance / increment))
    print("Subsampling distance " + str(distance) + " into " + str(increments) + " increments")
    if increments <= 1:
        end_point.sequence = current_sequence
        return [end_point]
    else:
        new_points = []
        for index in range(1, increments):
            percent = float(index) / float(increments)
            # Interpolate between the noisy "real" positions
            interpolated_position = interpolate_joints(start_point.position_desired, end_point.position_desired, percent)
            [secs, nsecs] = interpolate_timesteps(start_point, end_point, percent)
            new_state = XTF.XTFState(interpolated_position, [], [], [], [], [], current_sequence + len(new_points), (secs, nsecs))
            # Interpolate between the exact "clean" reference positions to compute expected cost
            interpolated_reference_position = interpolate_joints(start_point.extras["reference_position"], end_point.extras["reference_position"], percent)
            new_state.extras["reference_position"] = interpolated_reference_position
            new_points.append(new_state)
        end_point.sequence = current_sequence + len(new_points)
        new_points.append(end_point)
        return new_points

def euclidean_distance(p1, p2):
    dist = 0.0
    assert(len(p1) == len(p2))
    for [pp1, pp2] in zip(p1, p2):
        dist += ((pp2 - pp1) ** 2)
    return math.sqrt(dist)

def interpolate_timesteps(start, end, percent):
    start_secs = rospy.Time(start.secs, start.nsecs).to_sec()
    end_secs = rospy.Time(end.secs, end.nsecs).to_sec()
    new_time = rospy.Time(start_secs + ((end_secs - start_secs) * percent))
    return [new_time.secs, new_time.nsecs]

def interpolate(a, b, percent):
    return a + ((b - a) * percent)

def interpolate_joints(p1, p2, percent):
    interpolated = []
    assert(len(p1) == len(p2))
    for [pp1, pp2] in zip(p1, p2):
        interpolated.append(interpolate(pp1, pp2, percent))
    return interpolated

if __name__ == '__main__':
    subsample_trajectories("/home/calderpg/Dropbox/ROS_workspace/src/Research/baxter_uncertainty/data/planned_test_trajectories/")