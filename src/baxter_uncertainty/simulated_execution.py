#!/usr/bin/python

# Copyright (c) 2013, Rethink Robotics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the Rethink Robotics nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Baxter RSDK Joint Trajectory Action Server
"""
import bisect
from copy import deepcopy
import math
import time
import operator
from __builtin__ import xrange

import rospy

import actionlib

from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryFeedback,
    FollowJointTrajectoryResult,
)
from std_msgs.msg import (
    UInt16,
)
from trajectory_msgs.msg import (
    JointTrajectoryPoint,
)

import baxter_control
import baxter_dataflow
import baxter_interface

# Add XTF to provide logging support
import xtf.xtf as XTF
import deformable_astar.grc as GRC
from baxter_uncertainty.srv import *
import numpy
import random
from sensor_msgs.msg import JointState
import os


class SimulatedExecution(object):

    def __init__(self, limb, rate=100.0):
        self._control_rate = rate
        self._action_name = rospy.get_name()
        if limb == "left":
            self._joint_names = ["left_s0", "left_s1", "left_e0", "left_e1", "left_w0", "left_w1", "left_w2"]
        elif limb == "right":
            self._joint_names = ["right_s0", "right_s1", "right_e0", "right_e1", "right_w0", "right_w1", "right_w2"]

        # Initialize the current state
        self._current_position = []
        for idx in range(len(self._joint_names)):
            self._current_position.append(0.0)

        # Set up joint state feedback
        #self._state_sub = rospy.Subscriber("joint_states", JointState, self._state_cb)

        # Make a parser for XTF trajectories
        self._parser = XTF.XTFParser()

        # Make a GradientRejectionController
        self._grcontroller = GRC.GradientRejectionController()

        # Set the bound used by the Gradient Rejection Controller
        self._grc_control_factor = 30.0

        # Set the gain to use when computing corrections
        self._grc_gain = 0.25

        # Set the uncertainty params for simulated actuation
        self._minimum_variance = 0.0005
        self._variance_scaling = 0.0005

        # Setup the fake command publisher
        self._command_pub = rospy.Publisher('joint_commands', JointState)

        # Set the amount to "overtime" each step of a trajectory - we do this to give Baxter an easier time
        self._overtime_multiplier = 1.0

        # Set the location for log files
        self._in_log_location = "/home/calderpg/Dropbox/ROS_workspace/src/Research/baxter_uncertainty/data/planned_test_trajectories/"
        self._out_log_location = "/home/calderpg/Dropbox/ROS_workspace/src/Research/baxter_uncertainty/data/evaluated_trajectories/"

        # Set the correction mode
        #self._control_modes = {"point_neighbors": ComputeGradientRequest.POINT_NEIGHBORS, "gradient_blacklist": ComputeGradientRequest.GRADIENT_BLACKLIST, "sampled_eight_connected": ComputeGradientRequest.SAMPLED_EIGHT_CONNECTED, "hybrid_four_connected": ComputeGradientRequest.HYBRID_FOUR_CONNECTED, "four_connected": ComputeGradientRequest.FOUR_CONNECTED}
        #self._control_modes = {"gradient_blacklist": ComputeGradientRequest.GRADIENT_BLACKLIST}
        self._control_modes = {"sampled_eight_connected": ComputeGradientRequest.SAMPLED_EIGHT_CONNECTED, "hybrid_four_connected": ComputeGradientRequest.HYBRID_FOUR_CONNECTED}

        # Make a client to query the feature server
        self._feature_client = rospy.ServiceProxy("compute_baxter_cost_features", ComputeFeatures, persistent=True)

        # Make a client to query the uncertainty gradient server
        self._gradient_client = rospy.ServiceProxy("compute_baxter_cost_uncertainty_gradient", ComputeGradient, persistent=True)

        # Get the list of trajectory files
        file_list = os.listdir(self._in_log_location)
        xtf_files = []
        for filename in file_list:
            if "subsampled.xtf" in filename:
                xtf_files.append(filename)

        print("Evaluating " + str(len(xtf_files)) + " trajectory files...")
        # For each file, evaluate for each control mode
        for filename in xtf_files:
            print("Evaluating file: " + filename + "...")
            planned_trajectory = self._parser.ParseTraj(self._in_log_location + filename)
            for mode_name, mode_type in self._control_modes.iteritems():
                print("Evaluating with mode: " + mode_name)
                trajectory_copy = deepcopy(planned_trajectory)
                # Evaluate the trajectory
                trajectory_copy = self._evaluate_trajectory(trajectory_copy, mode_type)
                # Save the trajectory
                trajectory_path = self._out_log_location + mode_name + "/" + trajectory_copy.uid
                self._parser.ExportTraj(trajectory_copy, trajectory_path)
        print("...evaluation complete")

    def _state_cb(self, msg):
        new_state = []
        for name in self._joint_names:
            new_state.append(msg.position[msg.name.index(name)])
        self._current_position = new_state

    def _call_feature_client_safe(self, req, max_tries=5):
        try:
            return self._feature_client.call(req)
        except:
            if max_tries == 0:
                raise AttributeError("Feature client cannot connect")
            self._feature_client = rospy.ServiceProxy("compute_baxter_cost_features", ComputeFeatures, persistent=True)
            return self._call_feature_client_safe(req, (max_tries - 1))

    def _call_gradient_client_safe(self, req, max_tries=5):
        try:
            return self._gradient_client.call(req)
        except:
            if max_tries == 0:
                raise AttributeError("Gradient client cannot connect")
            self._gradient_client = rospy.ServiceProxy("compute_baxter_cost_uncertainty_gradient", ComputeGradient, persistent=True)
            return self._call_gradient_client_safe(req, (max_tries - 1))

    def _get_current_position(self, joint_names):
        temp_copy = deepcopy(self._current_position)
        ordered_config = []
        for joint_name in joint_names:
            local_index = self._joint_names.index(joint_name)
            ordered_config.append(temp_copy[local_index])
        return ordered_config

    def _interpolate(self, a, b, percent):
        return a + ((b - a) * percent)

    def _interpolate_joints(self, p1, p2, percent):
        interpolated = []
        assert(len(p1) == len(p2))
        for [pp1, pp2] in zip(p1, p2):
            interpolated.append(self._interpolate(pp1, pp2, percent))
        return interpolated

    def _sample_resultant_state(self, current_configuration, target_configuration):
        return target_configuration
        # Disabled since we've already added noise
        '''
        # For each joint, sample a new joint value given the current state and control input (target-current)
        # Determine the control input
        control_input = [0.0 for idx in xrange(len(current_configuration))]
        for index in range(len(current_configuration)):
            control_input[index] = target_configuration[index] - current_configuration[index]
        # Sample a new configuration based on the control input
        sampled_configuration = [0.0 for idx in xrange(len(current_configuration))]
        for index in range(len(current_configuration)):
            sampled_configuration[index] = self._sample_joint(current_configuration[index], control_input[index])
        return sampled_configuration
        '''

    def _sample_joint(self, current_value, control_input):
        mean = current_value + control_input
        variance = (self._variance_scaling * control_input) + self._minimum_variance
        sigma = math.sqrt(abs(variance))
        return random.gauss(mean, sigma)

    def _execute_to_state(self, joint_names, start_configuration, target_configuration, exec_time):
        # Overtime the current step
        exec_time *= self._overtime_multiplier
        # Check the sanity of the commanded point
        if len(joint_names) != len(target_configuration):
            rospy.logerr("%s: Commanded point and joint names do not match - aborting" % (self._action_name,))
            self._server.set_aborted()
            return False
        # Check the sanity of the commanded execution time
        if exec_time == 0.0:
            rospy.logerr("%s: Execution time is infeasible - skipping" % (self._action_name,))
            return True
        # Debug print
        print("Going to new state " + str(target_configuration) + " for " + str(exec_time) + " seconds")
        # Now that we think the point is safe to execute, let's do it
        control_duration = 1.0 / self._control_rate
        # Loop until exec_time runs out
        start_time = rospy.get_time()
        elapsed_time = rospy.get_time() - start_time
        while elapsed_time <= (exec_time + 2 * control_duration):
            # Interpolate a current state from start and target configuration
            percent = elapsed_time / exec_time
            target_point = self._interpolate_joints(start_configuration, target_configuration, percent)
            # As a shortcut, just overwrite the internal state
            self._current_position = target_point
            ## Execute to the current target
            #self._command_to_state(joint_names, target_point)
            ## Wait for the rest of the time step
            #control_rate.sleep()
            # Update time
            elapsed_time = rospy.get_time() - start_time
        return True

    def _command_to_state(self, joint_names, joint_values):
        command_msg = JointState()
        command_msg.header.stamp = rospy.get_rostime()
        command_msg.name = joint_names
        command_msg.position = joint_values
        self._command_pub.publish(command_msg)

    def _evaluate_trajectory(self, xtf_trajectory, correction_mode):
        if all("left" in jn for jn in xtf_trajectory.joint_names):
            side = "left"
        elif all("right" in jn for jn in xtf_trajectory.joint_names):
            side = "right"
        # Display the trajectory
        #raw_input("Run through sampled positions and compute executed cost...")
        for state in xtf_trajectory.trajectory:
            req = ComputeFeaturesRequest()
            req.ArmOption = ComputeFeaturesRequest.LEFT_ARM_ONLY
            req.FeatureOption = ComputeFeaturesRequest.COST_ONLY
            req.LeftArmConfiguration = state.position_desired
            req.GradientMultiplier = 0.1
            res = self._call_feature_client_safe(req)
        #raw_input("Simulate execution...")
        # Get the start state
        start_state = xtf_trajectory.trajectory[0]
        ################################################################################################################
        # Actually execute the trajectory
        ################################################################################################################
        # Execute to the first state of the trajectory
        target_point = start_state.position_desired
        exec_time = rospy.Duration(start_state.secs, start_state.nsecs).to_sec()
        if side == "left":
            ordered_joint_names = ["left_s0", "left_s1", "left_e0", "left_e1", "left_w0", "left_w1", "left_w2"]
        elif side == "right":
            ordered_joint_names = ["right_s0", "right_s1", "right_e0", "right_e1", "right_w0", "right_w1", "right_w2"]
        continue_execution = self._execute_to_state(ordered_joint_names, target_point, target_point, exec_time)
        if not continue_execution:
            rospy.logerr("%s: Start state execution failed, aborting" % (self._action_name,))
        # Update feedback and get the current state
        current_point = target_point
        # Get the cost of the current (real) point
        if side == "left":
            req = ComputeFeaturesRequest()
            req.ArmOption = ComputeFeaturesRequest.LEFT_ARM_ONLY
            req.FeatureOption = ComputeFeaturesRequest.COST_ONLY
            req.LeftArmConfiguration = current_point
            req.GradientMultiplier = 0.1
            res = self._call_feature_client_safe(req)
        elif side == "right":
            req = ComputeFeaturesRequest()
            req.ArmOption = ComputeFeaturesRequest.RIGHT_ARM_ONLY
            req.FeatureOption = ComputeFeaturesRequest.COST_ONLY
            req.RightArmConfiguration = current_point
            req.GradientMultiplier = 0.1
            res = self._call_feature_client_safe(req)
        if side == "left":
            start_state.extras["executed_cost"] = res.LeftArmCost
        elif side == "right":
            start_state.extras["executed_cost"] = res.RightArmCost
        start_state.position_actual = current_point
        # Check to make sure execution is OK
        if continue_execution:
            # Loop through the trajectory points (point 2 to point n-1)
            for point_index in xrange(1, len(xtf_trajectory.trajectory) - 1):
                current_state = xtf_trajectory.trajectory[point_index]
                prev_state = xtf_trajectory.trajectory[point_index - 1]
                # Adjust the target as needed
                current_state.extras["use_grc"] = True
                if "use_grc" in current_state.extras.keys() and current_state.extras["use_grc"]:
                    # Get the next target position
                    next_target_position = xtf_trajectory.trajectory[point_index + 1].position_desired
                    # Get the current real position
                    current_real_position = prev_state.position_actual
                    # Get the current ideal position
                    current_ideal_position = prev_state.position_desired
                    # Get the current target position
                    current_target_position = current_state.position_desired
                    # Get the current cost+uncertainty gradient
                    if side == "left":
                        req = ComputeGradientRequest()
                        req.ArmGradientOption = ComputeGradientRequest.LEFT_ARM_ONLY
                        req.ControlGenerationMode = correction_mode
                        req.MaxControlsToCheck = 26
                        req.ExpectedCostSamples = 100
                        req.LeftArmConfiguration = current_real_position
                        req.GradientMultiplier = self._grc_gain
                        res = self._call_gradient_client_safe(req)
                        current_gradient = res.LeftArmGradient
                    elif side == "right":
                        req = ComputeGradientRequest()
                        req.ArmGradientOption = ComputeGradientRequest.RIGHT_ARM_ONLY
                        req.ControlGenerationMode = correction_mode
                        req.MaxControlsToCheck = 26
                        req.ExpectedCostSamples = 100
                        req.RightArmConfiguration = current_real_position
                        req.GradientMultiplier = self._grc_gain
                        res = self._call_gradient_client_safe(req)
                        current_gradient = res.RightArmGradient
                    # Check if we've gotten an empty gradient
                    if len(current_gradient) != 7:
                        current_gradient = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    # Make sure the current gradient is not NAN
                    if any(math.isnan(val) for val in current_gradient):
                        rospy.logwarn("%s: Current gradient is NaN, setting to zero for safety" % (self._action_name,))
                        current_gradient = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    # Actually call the GRC
                    corrected_target = self._grcontroller.compute_new_target(numpy.array(current_ideal_position), numpy.array(current_real_position), numpy.array(current_target_position), numpy.array(next_target_position), numpy.array(current_gradient), self._grc_control_factor)
                    # Make sure the corrected target is back to 'real' Python and not numpy
                    grc_corrected_target = []
                    for value in corrected_target:
                        grc_corrected_target.append(float(value))
                else:
                    # GRC is disabled, go to the original target point
                    grc_corrected_target = current_state.position_desired
                # Execute the current step
                target_point = grc_corrected_target
                exec_time = rospy.Duration(current_state.secs, current_state.nsecs).to_sec() - rospy.Duration(prev_state.secs, prev_state.nsecs).to_sec()
                previous_point = prev_state.position_actual
                continue_execution = self._execute_to_state(ordered_joint_names, previous_point, target_point, exec_time)
                if not continue_execution:
                    rospy.logerr("%s: Middle state execution failed, aborting" % (self._action_name,))
                # Update feedback and get the current state
                current_point = target_point
                # Get the cost of the current (real) point
                if side == "left":
                    req = ComputeFeaturesRequest()
                    req.ArmOption = ComputeFeaturesRequest.LEFT_ARM_ONLY
                    req.FeatureOption = ComputeFeaturesRequest.COST_ONLY
                    req.LeftArmConfiguration = current_point
                    req.GradientMultiplier = 0.1
                    #raw_input("Real cost check request: " + str(req) + "...")
                    res = self._call_feature_client_safe(req)
                elif side == "right":
                    req = ComputeFeaturesRequest()
                    req.ArmOption = ComputeFeaturesRequest.RIGHT_ARM_ONLY
                    req.FeatureOption = ComputeFeaturesRequest.COST_ONLY
                    req.RightArmConfiguration = current_point
                    req.GradientMultiplier = 0.1
                    res = self._call_feature_client_safe(req)
                if side == "left":
                    current_state.extras["executed_cost"] = res.LeftArmCost
                elif side == "right":
                    current_state.extras["executed_cost"] = res.RightArmCost
                current_state.position_actual = current_point
                # Stop looping if execution has failed
                if not continue_execution:
                    break
            # Execute to the final state of the trajectory
            if continue_execution:
                end_state = xtf_trajectory.trajectory[-1]
                prev_state = xtf_trajectory.trajectory[-2]
                target_point = end_state.position_desired
                exec_time = rospy.Duration(end_state.secs, end_state.nsecs).to_sec() - rospy.Duration(prev_state.secs, prev_state.nsecs).to_sec()
                previous_point = prev_state.position_actual
                continue_execution = self._execute_to_state(ordered_joint_names, previous_point, target_point, exec_time)
                if not continue_execution:
                    rospy.logerr("%s: End state execution failed, aborting" % (self._action_name,))
                # Update feedback and get the current state
                current_point = target_point
                # Get the cost of the current (real) point
                if side == "left":
                    req = ComputeFeaturesRequest()
                    req.ArmOption = ComputeFeaturesRequest.LEFT_ARM_ONLY
                    req.FeatureOption = ComputeFeaturesRequest.COST_ONLY
                    req.LeftArmConfiguration = current_point
                    req.GradientMultiplier = 0.1
                    res = self._call_feature_client_safe(req)
                elif side == "right":
                    req = ComputeFeaturesRequest()
                    req.ArmOption = ComputeFeaturesRequest.RIGHT_ARM_ONLY
                    req.FeatureOption = ComputeFeaturesRequest.COST_ONLY
                    req.RightArmConfiguration = current_point
                    req.GradientMultiplier = 0.1
                    res = self._call_feature_client_safe(req)
                if side == "left":
                    end_state.extras["executed_cost"] = res.LeftArmCost
                elif side == "right":
                    end_state.extras["executed_cost"] = res.RightArmCost
                end_state.position_actual = current_point
            else:
                # Execution has been halted due to an error
                rospy.logerr("%s: Aborting further execution" % (self._action_name,))
        else:
            # Execution has been halted due to an error
            rospy.logerr("%s: Aborting further execution" % (self._action_name,))
        ################################################################################################################
        return xtf_trajectory

if __name__ == '__main__':
    rospy.init_node("simulated_execution")
    SimulatedExecution("left")