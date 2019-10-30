#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <random>
#include <ros/ros.h>
#include "baxter_uncertainty/baxter_uncertainty_model.hpp"

#define MINIMUM_VARIANCE 0.001
#define VARIANCE_SCALING 0.001

using namespace BaxterUncertainty;

std::vector<double> BaxterJointUncertaintyModel::SampleLeftArm(std::vector<double>& configuration, std::vector<double>& control_input)
{
    //ROS_INFO("Building distributions to sample from");
    // Make the normal distributions for each joint
    std::normal_distribution<double> joint1_distribution(configuration[0] + control_input[0], (control_input[0] * VARIANCE_SCALING) + MINIMUM_VARIANCE);
    std::normal_distribution<double> joint2_distribution(configuration[1] + control_input[1], (control_input[1] * VARIANCE_SCALING) + MINIMUM_VARIANCE);
    std::normal_distribution<double> joint3_distribution(configuration[2] + control_input[2], (control_input[2] * VARIANCE_SCALING) + MINIMUM_VARIANCE);
    std::normal_distribution<double> joint4_distribution(configuration[3] + control_input[3], (control_input[3] * VARIANCE_SCALING) + MINIMUM_VARIANCE);
    std::normal_distribution<double> joint5_distribution(configuration[4] + control_input[4], (control_input[4] * VARIANCE_SCALING) + MINIMUM_VARIANCE);
    std::normal_distribution<double> joint6_distribution(configuration[5] + control_input[5], (control_input[5] * VARIANCE_SCALING) + MINIMUM_VARIANCE);
    std::normal_distribution<double> joint7_distribution(configuration[6] + control_input[6], (control_input[6] * VARIANCE_SCALING) + MINIMUM_VARIANCE);
    //ROS_INFO("Sampling from the distributions");
    // Sample from them
    std::vector<double> sampled_configuration(7);
    sampled_configuration[0] = joint1_distribution(PRNG_);
    sampled_configuration[1] = joint2_distribution(PRNG_);
    sampled_configuration[2] = joint3_distribution(PRNG_);
    sampled_configuration[3] = joint4_distribution(PRNG_);
    sampled_configuration[4] = joint5_distribution(PRNG_);
    sampled_configuration[5] = joint6_distribution(PRNG_);
    sampled_configuration[6] = joint7_distribution(PRNG_);
    //ROS_INFO("Sampling complete");
    return sampled_configuration;
}

std::vector<double> BaxterJointUncertaintyModel::SampleRightArm(std::vector<double>& configuration, std::vector<double>& control_input)
{
    //ROS_INFO("Building distributions to sample from");
    // Make the normal distributions for each joint
    std::normal_distribution<double> joint1_distribution(configuration[0] + control_input[0], (control_input[0] * VARIANCE_SCALING) + MINIMUM_VARIANCE);
    std::normal_distribution<double> joint2_distribution(configuration[1] + control_input[1], (control_input[1] * VARIANCE_SCALING) + MINIMUM_VARIANCE);
    std::normal_distribution<double> joint3_distribution(configuration[2] + control_input[2], (control_input[2] * VARIANCE_SCALING) + MINIMUM_VARIANCE);
    std::normal_distribution<double> joint4_distribution(configuration[3] + control_input[3], (control_input[3] * VARIANCE_SCALING) + MINIMUM_VARIANCE);
    std::normal_distribution<double> joint5_distribution(configuration[4] + control_input[4], (control_input[4] * VARIANCE_SCALING) + MINIMUM_VARIANCE);
    std::normal_distribution<double> joint6_distribution(configuration[5] + control_input[5], (control_input[5] * VARIANCE_SCALING) + MINIMUM_VARIANCE);
    std::normal_distribution<double> joint7_distribution(configuration[6] + control_input[6], (control_input[6] * VARIANCE_SCALING) + MINIMUM_VARIANCE);
    //ROS_INFO("Sampling from the distributions");
    // Sample from them
    std::vector<double> sampled_configuration(7);
    sampled_configuration[0] = joint1_distribution(PRNG_);
    sampled_configuration[1] = joint2_distribution(PRNG_);
    sampled_configuration[2] = joint3_distribution(PRNG_);
    sampled_configuration[3] = joint4_distribution(PRNG_);
    sampled_configuration[4] = joint5_distribution(PRNG_);
    sampled_configuration[5] = joint6_distribution(PRNG_);
    sampled_configuration[6] = joint7_distribution(PRNG_);
    //ROS_INFO("Sampling complete");
    return sampled_configuration;
}

