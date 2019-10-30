#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <random>
#include <Eigen/Geometry>
#include <time.h>
#include <chrono>
#include <ros/ros.h>
#include <sdf_tools/sdf_builder.hpp>
#include "baxter_uncertainty/fast_baxter_forward_kinematics.hpp"
#include "baxter_uncertainty/baxter_uncertainty_model.hpp"
#include "baxter_uncertainty/ComputeFeatures.h"
#include "baxter_uncertainty/ComputeGradient.h"
#include "baxter_uncertainty/prettyprint.h"
#include "baxter_uncertainty/eigen_pinv.hpp"

// Control value definitions
#define CONTROL_PRIMITIVE 0.1
#define JOINTS 7
#define DECAY_RATE 0.05
#define SDF_DISTANCE_THRESHOLD 0.20
#define RCOND 0.001

// Set if we only want to update the SDF at load
//#define DISABLE_SDF_RELOAD

// Set if all gradients from points outside of the SDF_DISTANCE_THRESHOLD should be ignored
#define ALWAYS_USE_SDF_DISTANCE_THRESHOLD

// Set which kinematics system to use (MoveIt or thread-safe matrix version)
//#define USE_FAST_KINEMATICS

// Set which type of Jacobian matrix operation is used. If not defined, this uses Jacobian transpose
#define USE_JACOBIAN_PSEUDOINVERSE

// Enable/disable debug display markers (disable to improve performance)
#define ENABLE_DEBUG_DISPLAY_MARKERS

// Enable "Press ENTER to continue..." in gradient blacklist mode
//#define ENABLE_INTERACTIVE_BLACKLISTING

// Enable verbose debug printing when computing features
//#define ENABLE_VERBOSE_FEATURES

// Enable "Press ENTER to continue..." when computing features
//#define ENABLE_INTERACTIVE_FEATURES

// Enable "Press ENTER to continue..." before returning a control input
//#define ENABLE_INTERACTIVE_CONTROL_INPUTS

enum CONTROL_GENERATION_MODES {FOURCONNECTED, HYBRIDFOURCONNECTED, SAMPLEDEIGHTCONNECTED, GRADIENTBLACKLIST, POINTNEIGHBORS, UNKNOWNMODE};

class StaticBaxterJointUncertaintyCostGradientComputer
{
protected:

    ros::NodeHandle nh_;
    BaxterUncertainty::BaxterJointUncertaintyModel model_;
    std::vector<std::vector<Eigen::Vector3d>> left_link_points_;
    std::vector<std::vector<Eigen::Vector3d>> right_link_points_;
    std::vector<std::string> baxter_left_arm_link_names_;
    std::vector<std::string> baxter_right_arm_link_names_;
    std::vector<Eigen::Vector3d> baxter_basic_link_offsets_;
    sdf_tools::SignedDistanceField sdf_;
    std::mt19937_64 PRNG_;
    // Storage for a model of Baxter
    robot_model::RobotModelPtr baxter_model_;
    robot_model::RobotStatePtr baxter_kinematic_state_;
    std::unique_ptr<robot_model::JointModelGroup> baxter_left_arm_group_;
    std::unique_ptr<robot_model::JointModelGroup> baxter_right_arm_group_;
    // Marker publishers
    ros::Publisher sdf_display_pub_;
    ros::Publisher point_display_pub_;

public:

    StaticBaxterJointUncertaintyCostGradientComputer(ros::NodeHandle& nh, std::mt19937_64& prng, std::string& left_arm_points_dir, std::string& right_arm_points_dir, std::string& sdf_filepath) : nh_(nh), PRNG_(prng), model_(PRNG_)
    {
        // Load link names and points for both arms
        baxter_left_arm_link_names_ = GetLeftArmLinkNames();
        left_link_points_ = LoadArmData(left_arm_points_dir, baxter_left_arm_link_names_);
        baxter_right_arm_link_names_ = GetRightArmLinkNames();
        right_link_points_ = LoadArmData(right_arm_points_dir, baxter_right_arm_link_names_);
        baxter_basic_link_offsets_ = GetBasicLinkOffsets();
        // Set up an internal robot model
        robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
        baxter_model_ = robot_model_loader.getModel();
        baxter_kinematic_state_ = robot_model::RobotStatePtr(new robot_state::RobotState(baxter_model_));
        baxter_kinematic_state_->setToDefaultValues();
        baxter_kinematic_state_->update();
        baxter_left_arm_group_ = std::unique_ptr<robot_model::JointModelGroup>(baxter_model_->getJointModelGroup("left_arm"));
        baxter_right_arm_group_ = std::unique_ptr<robot_model::JointModelGroup>(baxter_model_->getJointModelGroup("right_arm"));
        // Set up the display marker publishers
        sdf_display_pub_ = nh_.advertise<visualization_msgs::Marker>("sdf_markers", 1, true);
        point_display_pub_ = nh_.advertise<visualization_msgs::Marker>("point_markers", 1, true);
        // Load the sdf from file
        ImportInternalSDFFromFile(sdf_filepath);
        // Draw the SDF
        RedrawSignedDistanceField(0.05);
    }

    std::vector<std::string> GetLeftArmLinkNames()
    {
        std::vector<std::string> link_names(JOINTS);
        link_names[0] = "left_upper_shoulder";
        link_names[1] = "left_lower_shoulder";
        link_names[2] = "left_upper_elbow";
        link_names[3] = "left_lower_elbow";
        link_names[4] = "left_upper_forearm";
        link_names[5] = "left_lower_forearm";
        link_names[6] = "left_hand";
        return link_names;
    }

    std::vector<std::string> GetRightArmLinkNames()
    {
        std::vector<std::string> link_names(JOINTS);
        link_names[0] = "right_upper_shoulder";
        link_names[1] = "right_lower_shoulder";
        link_names[2] = "right_upper_elbow";
        link_names[3] = "right_lower_elbow";
        link_names[4] = "right_upper_forearm";
        link_names[5] = "right_lower_forearm";
        link_names[6] = "right_hand";
        return link_names;
    }

    std::vector<Eigen::Vector3d> GetBasicLinkOffsets()
    {
        std::vector<Eigen::Vector3d> link_offsets(JOINTS);
        link_offsets[0] = Eigen::Vector3d(0.0, 0.0, 0.0);
        link_offsets[1] = Eigen::Vector3d(0.0, 0.0, 0.0);
        link_offsets[2] = Eigen::Vector3d(0.0, 0.0, 0.0);
        link_offsets[3] = Eigen::Vector3d(0.0, 0.0, 0.0);
        link_offsets[4] = Eigen::Vector3d(0.0, 0.0, 0.0);
        link_offsets[5] = Eigen::Vector3d(0.0, 0.0, 0.0);
        link_offsets[6] = Eigen::Vector3d(0.0, 0.0, 0.0);
        return link_offsets;
    }

    std::vector<std::vector<Eigen::Vector3d> > LoadArmData(std::string& arm_points_dir, std::vector<std::string>& arm_link_names)
    {
        std::vector<std::vector<Eigen::Vector3d>> points(JOINTS);
        for (size_t link_idx = 0; link_idx < JOINTS; link_idx++)
        {
            std::string full_filename = arm_points_dir + "/" + arm_link_names[link_idx] + ".points";
            FILE* link_points_file = fopen(full_filename.c_str(), "r");
            if (link_points_file == NULL)
            {
                ROS_FATAL("Data file not found - %s", full_filename.c_str());
                exit(1);
            }
            // Read the first link header
            int num_points = 0;
            int read = fscanf(link_points_file, "Robot link point data file - %d points", &num_points);
            if (read != 1)
            {
                ROS_FATAL("Invalid data file for link (invalid header) - %s", full_filename.c_str());
                exit(1);
            }
            // Make the container
            std::vector<Eigen::Vector3d> link_points(num_points);
            // Read the file contents
            for (int point_idx = 0; point_idx < num_points; point_idx++)
            {
                double x, y, z;
                read = fscanf(link_points_file, "%lf, %lf, %lf\n", &x, &y, &z);
                if (read != 3)
                {
                    ROS_FATAL("Invalid data file for link (invalid line) - %s", full_filename.c_str());
                    exit(1);
                }
                link_points[point_idx] = Eigen::Vector3d(x, y, z);
            }
            // Close the file
            fclose(link_points_file);
            points[link_idx] = link_points;
        }
        return points;
    }

    visualization_msgs::Marker ExportSDFDisplayMarker(float alpha)
    {
        return sdf_.ExportForDisplay(alpha);
    }

    void RedrawSignedDistanceField(float alpha)
    {
        sdf_display_pub_.publish(ExportSDFDisplayMarker(alpha));
    }

    inline double LookupInSDF(Eigen::Vector3d& point, sdf_tools::SignedDistanceField& sdf)
    {
        return sdf.Get(point.x(), point.y(), point.z());
    }

    inline std::vector<double> LookupGradientInSDF(Eigen::Vector3d& point, sdf_tools::SignedDistanceField& sdf)
    {
        std::vector<double> point_gradient = sdf.GetGradient(point.x(), point.y(), point.z());
        if (point_gradient.size() == 3)
        {
            if (point_gradient[0] == INFINITY || point_gradient[1] == INFINITY || point_gradient[2] == INFINITY)
            {
                point_gradient = std::vector<double>();
            }
            else if (isnan(point_gradient[0]) || isnan(point_gradient[1]) || isnan(point_gradient[2]))
            {
                point_gradient = std::vector<double>();
            }
#ifdef ALWAYS_USE_SDF_DISTANCE_THRESHOLD
            if (LookupInSDF(point, sdf) > SDF_DISTANCE_THRESHOLD)
            {
                point_gradient = std::vector<double>();
            }
#endif
        }
        return point_gradient;
    }

    inline std::vector<Eigen::Affine3d> ComputeLeftArmLinkTransforms(std::vector<double>& current_configuration)
    {
#ifdef USE_FAST_KINEMATICS
        return FastBaxterForwardKinematics::ComputeLeftArmLinkTransforms(current_configuration);
#else
        // Compute link transforms using MoveIt's kinematics
        std::vector<Eigen::Affine3d> link_transforms(JOINTS);
        // Update joint values
        baxter_kinematic_state_->setJointGroupPositions(baxter_left_arm_group_.get(), current_configuration);
        // Update the joint transforms
        baxter_kinematic_state_->update(true);
        // Get the transforms for each link
        link_transforms[0] = baxter_kinematic_state_->getGlobalLinkTransform(baxter_left_arm_link_names_[0]);
        link_transforms[1] = baxter_kinematic_state_->getGlobalLinkTransform(baxter_left_arm_link_names_[1]);
        link_transforms[2] = baxter_kinematic_state_->getGlobalLinkTransform(baxter_left_arm_link_names_[2]);
        link_transforms[3] = baxter_kinematic_state_->getGlobalLinkTransform(baxter_left_arm_link_names_[3]);
        link_transforms[4] = baxter_kinematic_state_->getGlobalLinkTransform(baxter_left_arm_link_names_[4]);
        link_transforms[5] = baxter_kinematic_state_->getGlobalLinkTransform(baxter_left_arm_link_names_[5]);
        link_transforms[6] = baxter_kinematic_state_->getGlobalLinkTransform(baxter_left_arm_link_names_[6]);
        return link_transforms;
#endif
    }

    inline std::vector<Eigen::Affine3d> ComputeRightArmLinkTransforms(std::vector<double>& current_configuration)
    {
#ifdef USE_FAST_KINEMATICS
        return FastBaxterForwardKinematics::ComputeRightArmLinkTransforms(current_configuration);
#else
        // Compute link transforms using MoveIt's kinematics
        std::vector<Eigen::Affine3d> link_transforms(JOINTS);
        // Update joint values
        baxter_kinematic_state_->setJointGroupPositions(baxter_right_arm_group_.get(), current_configuration);
        // Update the joint transforms
        baxter_kinematic_state_->update(true);
        // Get the transforms for each link
        link_transforms[0] = baxter_kinematic_state_->getGlobalLinkTransform(baxter_right_arm_link_names_[0]);
        link_transforms[1] = baxter_kinematic_state_->getGlobalLinkTransform(baxter_right_arm_link_names_[1]);
        link_transforms[2] = baxter_kinematic_state_->getGlobalLinkTransform(baxter_right_arm_link_names_[2]);
        link_transforms[3] = baxter_kinematic_state_->getGlobalLinkTransform(baxter_right_arm_link_names_[3]);
        link_transforms[4] = baxter_kinematic_state_->getGlobalLinkTransform(baxter_right_arm_link_names_[4]);
        link_transforms[5] = baxter_kinematic_state_->getGlobalLinkTransform(baxter_right_arm_link_names_[5]);
        link_transforms[6] = baxter_kinematic_state_->getGlobalLinkTransform(baxter_right_arm_link_names_[6]);
        return link_transforms;
#endif
    }

    std::vector<double> EstimateBestLeftArmControlInput(std::vector<double>& current_configuration, CONTROL_GENERATION_MODES control_generation_mode, u_int32_t max_controls_to_check, u_int32_t expected_cost_samples, double gradient_multiplier, bool update_SDF)
    {
        // Get the current jacobian
        //baxter_kinematic_state_->setJointGroupPositions(baxter_left_arm_group_.get(), current_configuration);
        //Eigen::MatrixXd current_jacobian = baxter_kinematic_state_->getJacobian(baxter_left_arm_group_.get());
        // Make the container for <control, cost> pairs
        std::map<std::vector<double>, double> control_costs;
        ROS_INFO("Generating control inputs to evaluate");
        // Make the set of control inputs we want to consider
        std::vector<std::vector<double> > control_inputs = GenerateLeftArmControlInputs(control_generation_mode, current_configuration, max_controls_to_check, gradient_multiplier);
        ROS_INFO("Evaluating %lu control inputs", control_inputs.size());
        // Evaluate each of the control inputs
        for (size_t index = 0; index < control_inputs.size(); index++)
        {
            control_costs[control_inputs[index]] = ComputeExpectedLeftArmCost(current_configuration, control_inputs[index], expected_cost_samples);
        }
        // Pick the best of the controls
        ROS_INFO("Selecting best control input");
        //std::cout << "Generated control <inputs, cost>: " << PrettyPrint(control_costs, true) << std::endl;
        int evaluated = 0;
        double best_cost = -INFINITY;
        std::vector<double> best_control_input;
        std::map<std::vector<double>, double>::iterator itr;
        for (itr = control_costs.begin(); itr != control_costs.end(); ++itr)
        {
            if (itr->second > best_cost)
            {
                best_control_input = itr->first;
                best_cost = itr->second;
            }
            evaluated++;
        }
        ROS_INFO("Computed best control input (of %d options) with expected value %f as gradient", evaluated, best_cost);
        // Check to see if the best cost is infinity, if so, return zero
        if (best_cost == INFINITY)
        {
            ROS_WARN("Best cost is infinite, which means SDF contains no obstacles. Returning zero gradient");
            return std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        }
        return best_control_input;
    }

    std::vector<double> EstimateBestRightArmControlInput(std::vector<double>& current_configuration, CONTROL_GENERATION_MODES control_generation_mode, u_int32_t max_controls_to_check, u_int32_t expected_cost_samples, double gradient_multiplier, bool update_SDF=false)
    {
        // Get the current jacobian
        //baxter_kinematic_state_->setJointGroupPositions(baxter_right_arm_group_.get(), current_configuration);
        //Eigen::MatrixXd current_jacobian = baxter_kinematic_state_->getJacobian(baxter_right_arm_group_.get());
        // Make the container for <control, cost> pairs
        std::map<std::vector<double>, double> control_costs;
        ROS_INFO("Generating control inputs to evaluate");
        // Make the set of control inputs we want to consider
        std::vector<std::vector<double> > control_inputs = GenerateRightArmControlInputs(control_generation_mode, current_configuration, max_controls_to_check, gradient_multiplier);
        ROS_INFO("Evaluating %lu control inputs", control_inputs.size());
        // Evaluate each of the control inputs
        for (size_t index = 0; index < control_inputs.size(); index++)
        {
            control_costs[control_inputs[index]] = ComputeExpectedRightArmCost(current_configuration, control_inputs[index], expected_cost_samples);
        }
        // Pick the best of the controls
        ROS_INFO("Selecting best control input");
        //std::cout << "Generated control <inputs, cost>: " << PrettyPrint(control_costs, true) << std::endl;
        int evaluated = 0;
        double best_cost = -INFINITY;
        std::vector<double> best_control_input;
        std::map<std::vector<double>, double>::iterator itr;
        for (itr = control_costs.begin(); itr != control_costs.end(); ++itr)
        {
            if (itr->second > best_cost)
            {
                best_control_input = itr->first;
                best_cost = itr->second;
            }
            evaluated++;
        }
        ROS_INFO("Computed best control input (of %d options) with expected value %f as gradient", evaluated, best_cost);
        // Check to see if the best cost is infinity, if so, return zero
        if (best_cost == INFINITY)
        {
            ROS_WARN("Best cost is infinite, which means SDF contains no obstacles. Returning zero gradient");
            return std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        }
        return best_control_input;
    }

    std::vector<std::vector<double>> GenerateLeftArmControlInputs(CONTROL_GENERATION_MODES control_generation_mode, std::vector<double>& current_configuration, u_int32_t max_controls_to_check, double gradient_multiplier)
    {
        std::vector<std::vector<double>> control_inputs;
        // Generate a set of control options for the left arm depending on the generation mode
        switch (control_generation_mode)
        {
            case CONTROL_GENERATION_MODES::FOURCONNECTED:
                control_inputs = GenerateLeftArmControlInputs4Connected(current_configuration);
                break;
            case CONTROL_GENERATION_MODES::HYBRIDFOURCONNECTED:
                control_inputs = GenerateLeftArmControlInputsHybrid4Connected(current_configuration);
                break;
            case CONTROL_GENERATION_MODES::SAMPLEDEIGHTCONNECTED:
                control_inputs = GenerateLeftArmControlInputsSampled8Connected(current_configuration, max_controls_to_check);
                break;
            case CONTROL_GENERATION_MODES::GRADIENTBLACKLIST:
                control_inputs = GenerateLeftArmControlInputsFromGradient(current_configuration, max_controls_to_check, gradient_multiplier);
                break;
            case CONTROL_GENERATION_MODES::POINTNEIGHBORS:
                control_inputs = GenerateLeftArmControlInputsFromNeighbors(current_configuration, gradient_multiplier);
                break;
            case CONTROL_GENERATION_MODES::UNKNOWNMODE:
                ROS_ERROR("UNKNOWNMODE case");
                break;
            default:
                ROS_ERROR("WTF happened here?!");
        }
        // Return the generated inputs
        //std::cout << "Raw generated control inputs: " << PrettyPrint(control_inputs, true) << std::endl;
        return control_inputs;
    }

    std::vector<std::vector<double>> GenerateRightArmControlInputs(CONTROL_GENERATION_MODES control_generation_mode, std::vector<double>& current_configuration, u_int32_t max_controls_to_check, double gradient_multiplier)
    {
        std::vector<std::vector<double>> control_inputs;
        // Generate a set of control options for the right arm depending on the generation mode
        switch (control_generation_mode)
        {
            case CONTROL_GENERATION_MODES::FOURCONNECTED:
                control_inputs = GenerateRightArmControlInputs4Connected(current_configuration);
                break;
            case CONTROL_GENERATION_MODES::HYBRIDFOURCONNECTED:
                control_inputs = GenerateRightArmControlInputsHybrid4Connected(current_configuration);
                break;
            case CONTROL_GENERATION_MODES::SAMPLEDEIGHTCONNECTED:
                control_inputs = GenerateRightArmControlInputsSampled8Connected(current_configuration, max_controls_to_check);
                break;
            case CONTROL_GENERATION_MODES::GRADIENTBLACKLIST:
                control_inputs = GenerateRightArmControlInputsFromGradient(current_configuration, max_controls_to_check, gradient_multiplier);
                break;
            case CONTROL_GENERATION_MODES::POINTNEIGHBORS:
                control_inputs = GenerateRightArmControlInputsFromNeighbors(current_configuration, gradient_multiplier);
                break;
            case CONTROL_GENERATION_MODES::UNKNOWNMODE:
                ROS_ERROR("UNKNOWNMODE case");
                break;
            default:
                ROS_ERROR("WTF happened here?!");
        }
        // Return the generated inputs
        //std::cout << "Raw generated control inputs: " << PrettyPrint(control_inputs, true) << std::endl;
        return control_inputs;
    }

    std::vector<std::vector<double>> GenerateLeftArmControlInputs4Connected(std::vector<double>& current_configuration)
    {
        std::vector<double> joint_primitives {-CONTROL_PRIMITIVE, CONTROL_PRIMITIVE};
        size_t joint_primitives_size = joint_primitives.size();
        std::vector<std::vector<double> > control_options(JOINTS * joint_primitives_size);
        // Generate a set of control options for the left arm
        // Generate all 4-connected options
        for (size_t index = 0; index < control_options.size(); index++)
        {
            std::vector<double> control_option(JOINTS);
            size_t joint_index = index / joint_primitives_size;
            size_t primitive_index = index % joint_primitives_size;
            control_option[joint_index] = joint_primitives[primitive_index];
            control_options[index] = control_option;
        }
        return control_options;
    }

    std::vector<std::vector<double>> GenerateRightArmControlInputs4Connected(std::vector<double>& current_configuration)
    {
        std::vector<double> joint_primitives {-CONTROL_PRIMITIVE, CONTROL_PRIMITIVE};
        size_t joint_primitives_size = joint_primitives.size();
        std::vector<std::vector<double> > control_options(JOINTS * joint_primitives_size);
        // Generate a set of control options for the right arm
        // Generate all 4-connected options
        for (size_t index = 0; index < control_options.size(); index++)
        {
            std::vector<double> control_option(JOINTS);
            size_t joint_index = index / joint_primitives_size;
            size_t primitive_index = index % joint_primitives_size;
            control_option[joint_index] = joint_primitives[primitive_index];
            control_options[index] = control_option;
        }
        return control_options;
    }

    std::vector<std::vector<double>> GenerateLeftArmControlInputsHybrid4Connected(std::vector<double>& current_configuration)
    {
        // Generate a set of control options for the left arm
        // Generate all 4-connected options plus a 2^joint subset of the 8-connected set
        // Generate the set of 4-connected options
        std::vector<std::vector<double> > control_options = GenerateLeftArmControlInputs4Connected(current_configuration);
        // Generate the 2^joint subset of the 8-connected set
        std::vector<double> joint_primitives {-CONTROL_PRIMITIVE, CONTROL_PRIMITIVE};
        size_t joint_primitives_size = joint_primitives.size();
        for (size_t idxq1 = 0; idxq1 < joint_primitives_size; idxq1++)
        {
            for (size_t idxq2 = 0; idxq2 < joint_primitives_size; idxq2++)
            {
                for (size_t idxq3 = 0; idxq3 < joint_primitives_size; idxq3++)
                {
                    for (size_t idxq4 = 0; idxq4 < joint_primitives_size; idxq4++)
                    {
                        for (size_t idxq5 = 0; idxq5 < joint_primitives_size; idxq5++)
                        {
                            for (size_t idxq6 = 0; idxq6 < joint_primitives_size; idxq6++)
                            {
                                for (size_t idxq7 = 0; idxq7 < joint_primitives_size; idxq7++)
                                {
                                    std::vector<double> control_option {joint_primitives[idxq1], joint_primitives[idxq2], joint_primitives[idxq3], joint_primitives[idxq4], joint_primitives[idxq5], joint_primitives[idxq6], joint_primitives[idxq7]};
                                    control_options.push_back(control_option);
                                }
                            }
                        }
                    }
                }
            }
        }
        return control_options;
    }

    std::vector<std::vector<double>> GenerateRightArmControlInputsHybrid4Connected(std::vector<double>& current_configuration)
    {
        // Generate a set of control options for the right arm
        // Generate all 4-connected options plus a 2^joint subset of the 8-connected set
        // Generate the set of 4-connected options
        std::vector<std::vector<double> > control_options = GenerateRightArmControlInputs4Connected(current_configuration);
        // Generate the 2^joint subset of the 8-connected set
        std::vector<double> joint_primitives {-CONTROL_PRIMITIVE, CONTROL_PRIMITIVE};
        size_t joint_primitives_size = joint_primitives.size();
        for (size_t idxq1 = 0; idxq1 < joint_primitives_size; idxq1++)
        {
            for (size_t idxq2 = 0; idxq2 < joint_primitives_size; idxq2++)
            {
                for (size_t idxq3 = 0; idxq3 < joint_primitives_size; idxq3++)
                {
                    for (size_t idxq4 = 0; idxq4 < joint_primitives_size; idxq4++)
                    {
                        for (size_t idxq5 = 0; idxq5 < joint_primitives_size; idxq5++)
                        {
                            for (size_t idxq6 = 0; idxq6 < joint_primitives_size; idxq6++)
                            {
                                for (size_t idxq7 = 0; idxq7 < joint_primitives_size; idxq7++)
                                {
                                    std::vector<double> control_option {joint_primitives[idxq1], joint_primitives[idxq2], joint_primitives[idxq3], joint_primitives[idxq4], joint_primitives[idxq5], joint_primitives[idxq6], joint_primitives[idxq7]};
                                    control_options.push_back(control_option);
                                }
                            }
                        }
                    }
                }
            }
        }
        return control_options;
    }

    std::vector<std::vector<double>> GenerateLeftArmControlInputsSampled8Connected(std::vector<double>& current_configuration, u_int32_t max_controls_to_check)
    {
        std::vector<double> joint_primitives {-CONTROL_PRIMITIVE, 0.0, CONTROL_PRIMITIVE};
        std::vector<std::vector<double> > control_options;
        // Generate a set of control options for the left arm
        // Sample randomly from the 3^joints - 1 value set of 8-connected control inputs
        for (u_int32_t sample = 0; sample < max_controls_to_check; sample++)
        {
            std::vector<double> sampled_control(JOINTS);
            for (int joint = 0; joint < JOINTS; joint++)
            {
                std::shuffle(joint_primitives.begin(), joint_primitives.end(), PRNG_);
                sampled_control[joint] = joint_primitives[0];
            }
            control_options.push_back(sampled_control);
        }
        return control_options;
    }

    std::vector<std::vector<double>> GenerateRightArmControlInputsSampled8Connected(std::vector<double>& current_configuration, u_int32_t max_controls_to_check)
    {
        std::vector<double> joint_primitives {-CONTROL_PRIMITIVE, 0.0, CONTROL_PRIMITIVE};
        std::vector<std::vector<double> > control_options;
        // Generate a set of control options for the right arm
        // Sample randomly from the 3^joints - 1 value set of 8-connected control inputs
        for (u_int32_t sample = 0; sample < max_controls_to_check; sample++)
        {
            std::vector<double> sampled_control(JOINTS);
            for (int joint = 0; joint < JOINTS; joint++)
            {
                std::shuffle(joint_primitives.begin(), joint_primitives.end(), PRNG_);
                sampled_control[joint] = joint_primitives[0];
            }
            control_options.push_back(sampled_control);
        }
        return control_options;
    }

    std::vector<std::vector<double>> GenerateLeftArmControlInputsFromNeighbors(std::vector<double>& current_configuration, double neighbor_multiplier)
    {
        // Produce a set of 26 possible control inputs for the left arm, starting with the "best"
        // These inputs are produced by computing the neighbors to each point of the arm, ranking them in "best" to "worst" order,
        // and then buiding a full-arm joint-space gradient from the per-point vectors
        //
        // First step - go through every point for every link, and get a ranked vector of its neighbors
        // * This container deserves an explanation *
        // for each link[for each point[vector of neighbors[each neighbor has a unit vector and a distance value]]]
        // This is how you index into it:
        // container[link_index][point_index][neighbor_index].first gives you the neighbor vector
        // container[link_index][point_index][neighbor_index].second gives you the distance value associated with that vector
        std::vector<std::vector<std::vector<std::pair<Eigen::Vector3d, double>>>> link_point_ranked_neighbor_vectors;
        // Populate this abomination of a container
        // Get the transforms for the left arm
        std::vector<Eigen::Affine3d> transforms = ComputeLeftArmLinkTransforms(current_configuration);
        ROS_INFO("Generating ranked neighbors for each point on the left arm");
        // For each link of the left arm
        for (size_t link_idx = 0; link_idx < JOINTS; link_idx++)
        {
            // Make the current-level container
            std::vector<std::vector<std::pair<Eigen::Vector3d, double>>> current_link_point_ranked_neighbor_vectors;
            // For each point on the current link
            for (size_t point_idx = 0; point_idx < left_link_points_[link_idx].size(); point_idx++)
            {
                // Get the real location of the current point
                Eigen::Vector3d real_location = transforms[link_idx] * left_link_points_[link_idx][point_idx];
                // Get the ranked neighbor vectors for the current point
                std::vector<std::pair<Eigen::Vector3d, double>> ranked_neighbor_vectors = GetRankedNeighborVectors(real_location, sdf_);
                assert(ranked_neighbor_vectors.size() == 26);
                // Store the neighbors
                current_link_point_ranked_neighbor_vectors.push_back(ranked_neighbor_vectors);
            }
            link_point_ranked_neighbor_vectors.push_back(current_link_point_ranked_neighbor_vectors);
        }
        // Second step - make a joint-space equivalents for the arm for each of the 26 possible neighbors
        ROS_INFO("Generating joint-space equivalents for each set of ranked neighbors");
        std::vector<std::vector<double>> arm_neighbors(26);
        for (size_t arm_neighbor_idx = 0; arm_neighbor_idx < arm_neighbors.size(); arm_neighbor_idx++)
        {
            arm_neighbors[arm_neighbor_idx] = GenerateLeftArmNeighborFromLinkPointNeighbors(current_configuration, link_point_ranked_neighbor_vectors, arm_neighbor_idx, neighbor_multiplier);
        }
        return arm_neighbors;
    }

    std::vector<std::vector<double> > GenerateRightArmControlInputsFromNeighbors(std::vector<double>& current_configuration, double neighbor_multiplier)
    {
        // Produce a set of 26 possible control inputs for the right arm, starting with the "best"
        // These inputs are produced by computing the neighbors to each point of the arm, ranking them in "best" to "worst" order,
        // and then buiding a full-arm joint-space gradient from the per-point vectors
        //
        // First step - go through every point for every link, and get a ranked vector of its neighbors
        // * This container deserves an explanation *
        // for each link[for each point[vector of neighbors[each neighbor has a unit vector and a distance value]]]
        // This is how you index into it:
        // container[link_index][point_index][neighbor_index].first gives you the neighbor vector
        // container[link_index][point_index][neighbor_index].second gives you the distance value associated with that vector
        std::vector<std::vector<std::vector<std::pair<Eigen::Vector3d, double>>>> link_point_ranked_neighbor_vectors;
        // Populate this abomination of a container
        // Get the transforms for the left arm
        std::vector<Eigen::Affine3d> transforms = ComputeRightArmLinkTransforms(current_configuration);
        ROS_INFO("Generating ranked neighbors for each point on the right arm");
        // For each link of the left arm
        for (size_t link_idx = 0; link_idx < JOINTS; link_idx++)
        {
            // Make the current-level container
            std::vector<std::vector<std::pair<Eigen::Vector3d, double>>> current_link_point_ranked_neighbor_vectors;
            // For each point on the current link
            for (size_t point_idx = 0; point_idx < right_link_points_[link_idx].size(); point_idx++)
            {
                // Get the real location of the current point
                Eigen::Vector3d real_location = transforms[link_idx] * right_link_points_[link_idx][point_idx];
                // Get the ranked neighbor vectors for the current point
                std::vector<std::pair<Eigen::Vector3d, double>> ranked_neighbor_vectors = GetRankedNeighborVectors(real_location, sdf_);
                // Store the neighbors
                current_link_point_ranked_neighbor_vectors.push_back(ranked_neighbor_vectors);
            }
            link_point_ranked_neighbor_vectors.push_back(current_link_point_ranked_neighbor_vectors);
        }
        // Second step - make a joint-space equivalents for the arm for each of the 26 possible neighbors
        ROS_INFO("Generating joint-space equivalents for each set of ranked neighbors");
        std::vector<std::vector<double>> arm_neighbors(26);
        for (size_t arm_neighbor_idx = 0; arm_neighbor_idx < arm_neighbors.size(); arm_neighbor_idx++)
        {
            arm_neighbors[arm_neighbor_idx] = GenerateRightArmNeighborFromLinkPointNeighbors(current_configuration, link_point_ranked_neighbor_vectors, arm_neighbor_idx, neighbor_multiplier);
        }
        return arm_neighbors;
    }

    std::vector<std::pair<Eigen::Vector3d, double> > GetRankedNeighborVectors(Eigen::Vector3d& point, sdf_tools::SignedDistanceField& sdf)
    {
        //ROS_INFO("Generating ranked neighbors for current point");
        std::vector<std::pair<Eigen::Vector3d, double>> neighbor_vectors; /// <- There are 26 neighbors in 3d
        double step_size = sdf.GetResolution();
        std::vector<int64_t> point_index = sdf.LocationToGridIndex(point.x(), point.y(), point.z());
        // If the point is outside the SDF (i.e. the point index returned has no elements), we return 26 zero-magnitude and zero-distance neighbors
        if (point_index.size() != 3)
        {
            std::pair<Eigen::Vector3d, double> default_neighbor_vector_pair(Eigen::Vector3d(0.0, 0.0, 0.0), 0.0);
            neighbor_vectors.resize(26, default_neighbor_vector_pair);
            //ROS_WARN("Point outside SDF, returning default neighbors");
            return neighbor_vectors;
        }
        // If the point is inside the SDF, we make the neighbors
        else
        {
            int neighbor_index = 0;
            // Make each of the 26 neighbors
            for (int x_offset = -1; x_offset <= 1; x_offset++)
            {
                for (int y_offset = -1; y_offset <= 1; y_offset++)
                {
                    for (int z_offset = -1; z_offset <= 1; z_offset++)
                    {
                        // Check for the case where our "neighbor" is actually ourself
                        if (x_offset != 0 || y_offset != 0 || z_offset != 0)
                        {
                            //ROS_INFO("Evaluating neighbor %d", neighbor_index);
                            // Check if the neighbor value is inside the SDF
                            if (sdf.CheckInBounds(point_index[0] + x_offset, point_index[1] + y_offset, point_index[2] + z_offset))
                            {
                                double neighbor_vector_distance = sdf.Get(point_index[0] + x_offset, point_index[1] + y_offset, point_index[2] + z_offset);
                                std::pair<Eigen::Vector3d, double> current_neighbor_vector_pair(Eigen::Vector3d(double(x_offset) * step_size, double(y_offset) * step_size, double(z_offset) * step_size), neighbor_vector_distance);
                                neighbor_vectors.push_back(current_neighbor_vector_pair);
                            }
                            // If the neighbor is outside the SDF, then we save the default one instead
                            else
                            {
                                std::pair<Eigen::Vector3d, double> default_neighbor_vector_pair(Eigen::Vector3d(0.0, 0.0, 0.0), 0.0);
                                neighbor_vectors.push_back(default_neighbor_vector_pair);
                            }
                            neighbor_index++;
                        }
                    }
                }
            }
            std::sort(neighbor_vectors.begin(), neighbor_vectors.end(), [](const std::pair<Eigen::Vector3d, double>& lhs, const std::pair<Eigen::Vector3d, double>& rhs) {return lhs.second > rhs.second;});
            // Return the sorted vector
            //ROS_INFO("Returning sorted neighbor vectors");
            return neighbor_vectors;
        }
    }

    std::vector<double> GenerateLeftArmNeighborFromLinkPointNeighbors(std::vector<double>& current_configuration, std::vector<std::vector<std::vector<std::pair<Eigen::Vector3d, double>>>>& link_point_ranked_neighbor_vectors, size_t arm_neighbor_index, double neighbor_multiplier)
    {
        // Set the configuration of the robot model
        baxter_kinematic_state_->setJointGroupPositions(baxter_left_arm_group_.get(), current_configuration);
        // Compute the joint-space cost gradient for the left arm
        Eigen::Matrix<double, Eigen::Dynamic, JOINTS> arm_jacobians;
        Eigen::Matrix<double, Eigen::Dynamic, 1> arm_neighbors;
        // Go through and assemble jacobians + gradients
        for (size_t link_idx = 0; link_idx < JOINTS; link_idx++)
        {
            std::string current_link_name = baxter_left_arm_link_names_[link_idx];
            //ROS_INFO("Evaluating link %s", current_link_name.c_str());
            for (size_t point_idx = 0; point_idx < left_link_points_[link_idx].size(); point_idx++)
            {
                // Compute the 'active jacobian' at the current location
                /// We use the RobotState::getJacobian(group, link, point, jacobian)
                /// function here, which computes the jacobian wrt the given point
                /// on the given link of the given group. This is equivalent to the
                /// OpenRAVE function CalculateActiveJacobian
                //ROS_INFO("Computing 'active jacobian' for link: %s", current_link_name.c_str());
                Eigen::MatrixXd new_active_jacobian; /// <- Make matrix storage for the gradient
                bool success = baxter_kinematic_state_->getJacobian(baxter_left_arm_group_.get(), baxter_kinematic_state_->getLinkModel(current_link_name), left_link_points_[link_idx][point_idx], new_active_jacobian); /// <- Do magic here
                if (!success)
                {
                    ROS_ERROR("Computing 'active jacobian' for left arm failed");
                }
                else
                {
                    //ROS_INFO("Computed 'active jacobian' for the left arm as a %ld by %ld matrix", new_active_jacobian.rows(), new_active_jacobian.cols());
                }
                // Get the translation-only section of the jacobian (3x7)
                Eigen::Matrix<double, 3, JOINTS> translation_jacobian;
                translation_jacobian << new_active_jacobian.row(0),new_active_jacobian.row(1),new_active_jacobian.row(2);
                // Print the matrix to the screen
                //std::cout << translation_jacobian << std::endl;
                // Make a new, larger, matrix to store the extended jacobians
                Eigen::Matrix<double, Eigen::Dynamic, JOINTS> extended_arm_jacobians;
                extended_arm_jacobians.resize(arm_jacobians.rows() + 3, Eigen::NoChange);
                // Append the active jacobian to the matrix of jacobians
                extended_arm_jacobians << arm_jacobians,translation_jacobian;
                arm_jacobians = extended_arm_jacobians;
                // Get the "neighbor vector" for the current point
                Eigen::Vector3d neighbor_vector = link_point_ranked_neighbor_vectors[link_idx][point_idx][arm_neighbor_index].first;
                double neighbor_vector_distance = link_point_ranked_neighbor_vectors[link_idx][point_idx][arm_neighbor_index].second;
                // Weight the vector based on the distance
                Eigen::Vector3d weighted_neighbor_vector = WeightVectorByDistance(neighbor_vector, neighbor_vector_distance);
                // Make a new, larger, matrix to store the extended gradients
                Eigen::Matrix<double, Eigen::Dynamic, 1> extended_arm_neighbors;
                extended_arm_neighbors.resize(arm_neighbors.rows() + 3, Eigen::NoChange);
                // Append the gradient to the vector of gradients
                extended_arm_neighbors << arm_neighbors,weighted_neighbor_vector;
                arm_neighbors = extended_arm_neighbors;
            }
        }
        // Check if either matrix is empty, if so, return zero vector
        if (arm_jacobians.rows() == 0 || arm_jacobians.cols() == 0)
        {
            ROS_WARN("Arm jacobians matrix is empty, returning zero gradient");
            return std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        }
        if (arm_neighbors.rows() == 0 || arm_neighbors.cols() == 0)
        {
            ROS_WARN("Arm neighbors matrix is empty, returning zero gradient");
            return std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        }
        //ROS_INFO("Doing Jacobian magic");
        // Do the math
        Eigen::Matrix<double, JOINTS, 1> raw_joint_space_neighbor = ProcessJacobian(arm_jacobians) * arm_neighbors;
        double norm = raw_joint_space_neighbor.norm();
        if (fabs(norm) < RCOND)
        {
            norm = 1.0;
        }
        double inv_norm = 1.0 / norm;
        // Store the computed gradient
        //ROS_INFO("Extracting the joint-space neighbor");
        std::vector<double> joint_space_neighbor(JOINTS);
        joint_space_neighbor[0] = neighbor_multiplier * raw_joint_space_neighbor(0,0) * inv_norm;
        joint_space_neighbor[1] = neighbor_multiplier * raw_joint_space_neighbor(1,0) * inv_norm;
        joint_space_neighbor[2] = neighbor_multiplier * raw_joint_space_neighbor(2,0) * inv_norm;
        joint_space_neighbor[3] = neighbor_multiplier * raw_joint_space_neighbor(3,0) * inv_norm;
        joint_space_neighbor[4] = neighbor_multiplier * raw_joint_space_neighbor(4,0) * inv_norm;
        joint_space_neighbor[5] = neighbor_multiplier * raw_joint_space_neighbor(5,0) * inv_norm;
        joint_space_neighbor[6] = neighbor_multiplier * raw_joint_space_neighbor(6,0) * inv_norm;
        //std::cout << "Raw joint space neighbor: " << PrettyPrint(raw_joint_space_neighbor) << std::endl;
        //std::cout << "Scaled joint space neighbor: " << PrettyPrint(joint_space_neighbor) << std::endl;
        return joint_space_neighbor;
    }

    std::vector<double> GenerateRightArmNeighborFromLinkPointNeighbors(std::vector<double>& current_configuration, std::vector<std::vector<std::vector<std::pair<Eigen::Vector3d, double>>>>& link_point_ranked_neighbor_vectors, size_t arm_neighbor_index, double neighbor_multiplier)
    {
        // Set the configuration of the robot model
        baxter_kinematic_state_->setJointGroupPositions(baxter_right_arm_group_.get(), current_configuration);
        // Compute the joint-space cost gradient for the left arm
        Eigen::Matrix<double, Eigen::Dynamic, JOINTS> arm_jacobians;
        Eigen::Matrix<double, Eigen::Dynamic, 1> arm_neighbors;
        // Go through and assemble jacobians + gradients
        for (size_t link_idx = 0; link_idx < JOINTS; link_idx++)
        {
            std::string current_link_name = baxter_right_arm_link_names_[link_idx];
            //ROS_INFO("Evaluating link %s", current_link_name.c_str());
            for (size_t point_idx = 0; point_idx < right_link_points_[link_idx].size(); point_idx++)
            {
                // Compute the 'active jacobian' at the current location
                /// We use the RobotState::getJacobian(group, link, point, jacobian)
                /// function here, which computes the jacobian wrt the given point
                /// on the given link of the given group. This is equivalent to the
                /// OpenRAVE function CalculateActiveJacobian
                //ROS_INFO("Computing 'active jacobian' for link: %s", current_link_name.c_str());
                Eigen::MatrixXd new_active_jacobian; /// <- Make matrix storage for the gradient
                bool success = baxter_kinematic_state_->getJacobian(baxter_right_arm_group_.get(), baxter_kinematic_state_->getLinkModel(current_link_name), right_link_points_[link_idx][point_idx], new_active_jacobian); /// <- Do magic here
                if (!success)
                {
                    ROS_ERROR("Computing 'active jacobian' for left arm failed");
                }
                else
                {
                    //ROS_INFO("Computed 'active jacobian' for the left arm as a %ld by %ld matrix", new_active_jacobian.rows(), new_active_jacobian.cols());
                }
                // Get the translation-only section of the jacobian (3x7)
                Eigen::Matrix<double, 3, JOINTS> translation_jacobian;
                translation_jacobian << new_active_jacobian.row(0),new_active_jacobian.row(1),new_active_jacobian.row(2);
                // Print the matrix to the screen
                //std::cout << translation_jacobian << std::endl;
                // Make a new, larger, matrix to store the extended jacobians
                Eigen::Matrix<double, Eigen::Dynamic, JOINTS> extended_arm_jacobians;
                extended_arm_jacobians.resize(arm_jacobians.rows() + 3, Eigen::NoChange);
                // Append the active jacobian to the matrix of jacobians
                extended_arm_jacobians << arm_jacobians,translation_jacobian;
                arm_jacobians = extended_arm_jacobians;
                // Get the "neighbor vector" for the current point
                Eigen::Vector3d neighbor_vector = link_point_ranked_neighbor_vectors[link_idx][point_idx][arm_neighbor_index].first;
                double neighbor_vector_distance = link_point_ranked_neighbor_vectors[link_idx][point_idx][arm_neighbor_index].second;
                // Weight the vector based on the distance
                Eigen::Vector3d weighted_neighbor_vector = WeightVectorByDistance(neighbor_vector, neighbor_vector_distance);
                // Make a new, larger, matrix to store the extended gradients
                Eigen::Matrix<double, Eigen::Dynamic, 1> extended_arm_neighbors;
                extended_arm_neighbors.resize(arm_neighbors.rows() + 3, Eigen::NoChange);
                // Append the gradient to the vector of gradients
                extended_arm_neighbors << arm_neighbors,weighted_neighbor_vector;
                arm_neighbors = extended_arm_neighbors;
            }
        }
        // Check if either matrix is empty, if so, return zero vector
        if (arm_jacobians.rows() == 0 || arm_jacobians.cols() == 0)
        {
            ROS_WARN("Arm jacobians matrix is empty, returning zero gradient");
            return std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        }
        if (arm_neighbors.rows() == 0 || arm_neighbors.cols() == 0)
        {
            ROS_WARN("Arm neighbors matrix is empty, returning zero gradient");
            return std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        }
        //ROS_INFO("Doing Jacobian magic");
        // Do the math
        Eigen::Matrix<double, JOINTS, 1> raw_joint_space_neighbor = ProcessJacobian(arm_jacobians) * arm_neighbors;
        double norm = raw_joint_space_neighbor.norm();
        if (fabs(norm) < RCOND)
        {
            norm = 1.0;
        }
        double inv_norm = 1.0 / norm;
        // Store the computed gradient
        //ROS_INFO("Extracting the joint-space neighbor");
        std::vector<double> joint_space_neighbor(JOINTS);
        joint_space_neighbor[0] = neighbor_multiplier * raw_joint_space_neighbor(0,0) * inv_norm;
        joint_space_neighbor[1] = neighbor_multiplier * raw_joint_space_neighbor(1,0) * inv_norm;
        joint_space_neighbor[2] = neighbor_multiplier * raw_joint_space_neighbor(2,0) * inv_norm;
        joint_space_neighbor[3] = neighbor_multiplier * raw_joint_space_neighbor(3,0) * inv_norm;
        joint_space_neighbor[4] = neighbor_multiplier * raw_joint_space_neighbor(4,0) * inv_norm;
        joint_space_neighbor[5] = neighbor_multiplier * raw_joint_space_neighbor(5,0) * inv_norm;
        joint_space_neighbor[6] = neighbor_multiplier * raw_joint_space_neighbor(6,0) * inv_norm;
        //std::cout << "Raw joint space neighbor: " << PrettyPrint(raw_joint_space_neighbor) << std::endl;
        //std::cout << "Scaled joint space neighbor: " << PrettyPrint(joint_space_neighbor) << std::endl;
        return joint_space_neighbor;
    }

    inline Eigen::Vector3d WeightVectorByDistance(Eigen::Vector3d& raw_vector, double distance)
    {
        if (distance > SDF_DISTANCE_THRESHOLD)
        {
            //ROS_INFO("Ignoring neighbor with distance %f > SDF_DISTANCE_THRESHOLD", distance);
            return Eigen::Vector3d(0.0, 0.0, 0.0);
        }
        else
        {
            double weight = 1.0 + (fabs(SDF_DISTANCE_THRESHOLD - distance) / SDF_DISTANCE_THRESHOLD);
            //ROS_INFO("Weighting neighbor with distance %f by weight %f", distance, weight);
            return raw_vector * weight;
        }
    }

    std::vector<std::vector<double>> GenerateLeftArmControlInputsFromGradient(std::vector<double>& current_configuration, u_int32_t max_controls_to_check, double gradient_multiplier)
    {
        sdf_tools::SignedDistanceField working_sdf_copy = sdf_;
        std::vector<std::vector<double> > control_options;
        // Generate a set of control options for the left arm using the gradient from the SDF
        // for i = 1:max_controls_to_check
        //     get_gradient()
        //     adapt SDF to blacklist the current gradient
        // repeat
        for (u_int32_t index = 0; index < max_controls_to_check; index++)
        {
#ifdef ENABLE_INTERACTIVE_BLACKLISTING
            point_display_pub_.publish(DrawLeftArmPointGradients(current_configuration, working_sdf_copy));
#endif
            // Get the current gradient and a new copy of the SDF with penalty values inserted
            std::pair<std::vector<double>, sdf_tools::SignedDistanceField> result = ComputeAndBlacklistLeftArmCostGradient(current_configuration, working_sdf_copy, gradient_multiplier);
#ifdef ENABLE_INTERACTIVE_BLACKLISTING
            std::cout << "Computed gradient: " << PrettyPrint(result.first) << std::endl;
#endif
            // Save the gradient
            control_options.push_back(result.first);
            // Save the new SDF
            working_sdf_copy = result.second;
#ifdef ENABLE_INTERACTIVE_BLACKLISTING
            std::cout << "Press ENTER to continue..." << std::endl;
            getchar();
#endif
        }
        return control_options;
    }

    std::vector<std::vector<double>> GenerateRightArmControlInputsFromGradient(std::vector<double>& current_configuration, u_int32_t max_controls_to_check, double gradient_multiplier)
    {
        sdf_tools::SignedDistanceField working_sdf_copy = sdf_;
        std::vector<std::vector<double> > control_options;
        // Generate a set of control options for the right arm using the gradient from the SDF
        // for i = 1:max_controls_to_check
        //     get_gradient()
        //     adapt SDF to blacklist the current gradient
        // repeat
        for (u_int32_t index = 0; index < max_controls_to_check; index++)
        {
#ifdef ENABLE_INTERACTIVE_BLACKLISTING
            point_display_pub_.publish(DrawRightArmPointGradients(current_configuration, working_sdf_copy));
#endif
            // Get the current gradient and a new copy of the SDF with penalty values inserted
            std::pair<std::vector<double>, sdf_tools::SignedDistanceField> result = ComputeAndBlacklistRightArmCostGradient(current_configuration, working_sdf_copy, gradient_multiplier);
#ifdef ENABLE_INTERACTIVE_BLACKLISTING
            std::cout << "Computed gradient: " << PrettyPrint(result.first) << std::endl;
#endif
            // Save the gradient
            control_options.push_back(result.first);
            // Save the new SDF
            working_sdf_copy = result.second;
#ifdef ENABLE_INTERACTIVE_BLACKLISTING
            std::cout << "Press ENTER to continue..." << std::endl;
            getchar();
#endif
        }
        return control_options;
    }

    inline float ComputePenaltyValue(float current_value, float target_value)
    {
        // Using the current value of the cell in the SDF, and the gradient in its direction
        // We want to gradually bring the gradient to zero
        double difference = target_value - current_value;
        double new_value = current_value + (difference * DECAY_RATE);
        //ROS_INFO("Assigned new value %f from current value %f and target %f", new_value, current_value, target_value);
        return new_value;
    }

    void InsertPenaltyVoxel(sdf_tools::SignedDistanceField& reference_SDF, Eigen::Vector3d& location, Eigen::Vector3d& gradient, sdf_tools::SignedDistanceField& modified_SDF)
    {
        //ROS_INFO("Inserting penalty voxel for location %f,%f,%f", location.x(), location.y(), location.z());
        // Check if the gradient is zero
        if (fabs(gradient.x()) == 0.0 && fabs(gradient.y()) == 0.0 && fabs(gradient.z()) == 0.0)
        {
            //ROS_WARN("Zero gradient, no penalty value(s) to insert");
            return;
        }
        // Get the index of the current location
        std::vector<int64_t> current_index = modified_SDF.LocationToGridIndex(location.x(), location.y(), location.z());
        if (current_index.empty())
        {
            //ROS_WARN("Location outside of SDF, no penalty value(s) to insert");
            return;
        }
        // Adjust the x-axis values
        float new_xp_val = ComputePenaltyValue(modified_SDF.Get(current_index[0] + 1, current_index[1], current_index[2]), reference_SDF.Get(current_index[0] - 1, current_index[1], current_index[2]));
        float new_xn_val = ComputePenaltyValue(modified_SDF.Get(current_index[0] - 1, current_index[1], current_index[2]), reference_SDF.Get(current_index[0] + 1, current_index[1], current_index[2]));
        modified_SDF.Set(current_index[0] + 1, current_index[1], current_index[2], new_xp_val);
        modified_SDF.Set(current_index[0] - 1, current_index[1], current_index[2], new_xn_val);
        // Adjust the y-axis values
        float new_yp_val = ComputePenaltyValue(modified_SDF.Get(current_index[0], current_index[1] + 1, current_index[2]), reference_SDF.Get(current_index[0], current_index[1] - 1, current_index[2]));
        float new_yn_val = ComputePenaltyValue(modified_SDF.Get(current_index[0], current_index[1] - 1, current_index[2]), reference_SDF.Get(current_index[0], current_index[1] + 1, current_index[2]));
        modified_SDF.Set(current_index[0], current_index[1] + 1, current_index[2], new_yp_val);
        modified_SDF.Set(current_index[0], current_index[1] - 1, current_index[2], new_yn_val);
        // Adjust the z-axis values
        float new_zp_val = ComputePenaltyValue(modified_SDF.Get(current_index[0], current_index[1], current_index[2] + 1), reference_SDF.Get(current_index[0], current_index[1], current_index[2] - 1));
        float new_zn_val = ComputePenaltyValue(modified_SDF.Get(current_index[0], current_index[1], current_index[2] - 1), reference_SDF.Get(current_index[0], current_index[1], current_index[2] + 1));
        modified_SDF.Set(current_index[0], current_index[1], current_index[2] + 1, new_zp_val);
        modified_SDF.Set(current_index[0], current_index[1], current_index[2] - 1, new_zn_val);
        // Done!
        return;
    }

    std::pair<std::vector<double>, sdf_tools::SignedDistanceField> ComputeAndBlacklistLeftArmCostGradient(std::vector<double>& current_configuration, sdf_tools::SignedDistanceField& signed_distance_field, double gradient_multiplier)
    {
        // Make a copy of the SDF to modify (not used in the gradient computation!)
        sdf_tools::SignedDistanceField writable_sdf = signed_distance_field;
        // Compute the joint-space cost gradient for the left arm
        Eigen::Matrix<double, Eigen::Dynamic, JOINTS> arm_jacobians;
        Eigen::Matrix<double, Eigen::Dynamic, 1> arm_gradients;
        // Go through and assemble jacobians + gradients
        std::vector<Eigen::Affine3d> transforms = ComputeLeftArmLinkTransforms(current_configuration);
        for (size_t link_idx = 0; link_idx < JOINTS; link_idx++)
        {
            std::string current_link_name = baxter_left_arm_link_names_[link_idx];
            //ROS_INFO("Evaluating link %s", current_link_name.c_str());
            for (size_t point_idx = 0; point_idx < left_link_points_[link_idx].size(); point_idx++)
            {
                // Get the 3D cost gradient of the current point
                Eigen::Vector3d real_location = transforms[link_idx] * left_link_points_[link_idx][point_idx];
                std::vector<double> point_gradient = LookupGradientInSDF(real_location, signed_distance_field);
                // Make sure the gradient check was inside bounds
                if (point_gradient.size() == 3)
                {
                    // Compute the 'active jacobian' at the current location
                    /// We use the RobotState::getJacobian(group, link, point, jacobian)
                    /// function here, which computes the jacobian wrt the given point
                    /// on the given link of the given group. This is equivalent to the
                    /// OpenRAVE function CalculateActiveJacobian
                    //ROS_INFO("Computing 'active jacobian' for link: %s", current_link_name.c_str());
                    Eigen::MatrixXd new_active_jacobian; /// <- Make matrix storage for the gradient
                    bool success = baxter_kinematic_state_->getJacobian(baxter_left_arm_group_.get(), baxter_kinematic_state_->getLinkModel(current_link_name), left_link_points_[link_idx][point_idx], new_active_jacobian); /// <- Do magic here
                    if (!success)
                    {
                        ROS_ERROR("Computing 'active jacobian' for left arm failed");
                    }
                    else
                    {
                        //ROS_INFO("Computed 'active jacobian' for the left arm as a %ld by %ld matrix", new_active_jacobian.rows(), new_active_jacobian.cols());
                    }
                    // Get the translation-only section of the jacobian (3x7)
                    Eigen::Matrix<double, 3, JOINTS> translation_jacobian;
                    translation_jacobian << new_active_jacobian.row(0),new_active_jacobian.row(1),new_active_jacobian.row(2);
                    // Print the matrix to the screen
                    //std::cout << translation_jacobian << std::endl;
                    // Make a new, larger, matrix to store the extended jacobians
                    Eigen::Matrix<double, Eigen::Dynamic, JOINTS> extended_arm_jacobians;
                    extended_arm_jacobians.resize(arm_jacobians.rows() + 3, Eigen::NoChange);
                    // Append the active jacobian to the matrix of jacobians
                    extended_arm_jacobians << arm_jacobians,translation_jacobian;
                    arm_jacobians = extended_arm_jacobians;
                    // Make current gradient into a vector
                    Eigen::Vector3d new_arm_gradient(point_gradient[0], point_gradient[1], point_gradient[2]);
                    // Make a new, larger, matrix to store the extended gradients
                    Eigen::Matrix<double, Eigen::Dynamic, 1> extended_arm_gradients;
                    extended_arm_gradients.resize(arm_gradients.rows() + 3, Eigen::NoChange);
                    // Append the gradient to the vector of gradients
                    extended_arm_gradients << arm_gradients,new_arm_gradient;
                    arm_gradients = extended_arm_gradients;
                    // Blacklist the gradient
                    Eigen::Vector3d cur_gradient(point_gradient[0], point_gradient[1], point_gradient[2]);
                    InsertPenaltyVoxel(sdf_, real_location, cur_gradient, writable_sdf);
                }
            }
        }
        // Check if either matrix is empty, if so, return zero vector
        if (arm_jacobians.rows() == 0 || arm_jacobians.cols() == 0)
        {
            ROS_WARN("Arm jacobians matrix is empty, returning zero gradient");
            std::vector<double> joint_space_cost_gradient{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            return std::pair<std::vector<double>, sdf_tools::SignedDistanceField>(joint_space_cost_gradient, writable_sdf);
        }
        if (arm_gradients.rows() == 0 || arm_gradients.cols() == 0)
        {
            ROS_WARN("Arm neighbors matrix is empty, returning zero gradient");
            std::vector<double> joint_space_cost_gradient{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            return std::pair<std::vector<double>, sdf_tools::SignedDistanceField>(joint_space_cost_gradient, writable_sdf);
        }
        //ROS_INFO("Doing Jacobian magic");
        // Do the math
        Eigen::Matrix<double, JOINTS, 1> raw_joint_space_cost_gradient = ProcessJacobian(arm_jacobians) * arm_gradients;
        double inv_norm = 1.0 / raw_joint_space_cost_gradient.norm();
        if (fabs(inv_norm) < RCOND)
        {
            inv_norm = 1.0;
        }
        // Store the computed gradient
        //ROS_INFO("Extracting the joint-space cost gradient");
        std::vector<double> joint_space_cost_gradient(JOINTS);
        joint_space_cost_gradient[0] = gradient_multiplier * raw_joint_space_cost_gradient(0,0) * inv_norm;
        joint_space_cost_gradient[1] = gradient_multiplier * raw_joint_space_cost_gradient(1,0) * inv_norm;
        joint_space_cost_gradient[2] = gradient_multiplier * raw_joint_space_cost_gradient(2,0) * inv_norm;
        joint_space_cost_gradient[3] = gradient_multiplier * raw_joint_space_cost_gradient(3,0) * inv_norm;
        joint_space_cost_gradient[4] = gradient_multiplier * raw_joint_space_cost_gradient(4,0) * inv_norm;
        joint_space_cost_gradient[5] = gradient_multiplier * raw_joint_space_cost_gradient(5,0) * inv_norm;
        joint_space_cost_gradient[6] = gradient_multiplier * raw_joint_space_cost_gradient(6,0) * inv_norm;
        return std::pair<std::vector<double>, sdf_tools::SignedDistanceField>(joint_space_cost_gradient, writable_sdf);
    }

    std::pair<std::vector<double>, sdf_tools::SignedDistanceField> ComputeAndBlacklistRightArmCostGradient(std::vector<double>& current_configuration, sdf_tools::SignedDistanceField& signed_distance_field, double gradient_multiplier)
    {
        // Make a copy of the SDF to modify (not used in the gradient computation!)
        sdf_tools::SignedDistanceField writable_sdf = signed_distance_field;
        // Compute the joint-space cost gradient for the right arm
        Eigen::Matrix<double, Eigen::Dynamic, JOINTS> arm_jacobians;
        Eigen::Matrix<double, Eigen::Dynamic, 1> arm_gradients;
        // Go through and assemble jacobians + gradients
        std::vector<Eigen::Affine3d> transforms = ComputeRightArmLinkTransforms(current_configuration);
        for (size_t link_idx = 0; link_idx < JOINTS; link_idx++)
        {
            std::string current_link_name = baxter_right_arm_link_names_[link_idx];
            //ROS_INFO("Evaluating link %s", current_link_name.c_str());
            for (size_t point_idx = 0; point_idx < right_link_points_[link_idx].size(); point_idx++)
            {
                // Get the 3D cost gradient of the current point
                Eigen::Vector3d real_location = transforms[link_idx] * right_link_points_[link_idx][point_idx];
                std::vector<double> point_gradient = LookupGradientInSDF(real_location, signed_distance_field);
                // Make sure the gradient check was inside bounds
                if (point_gradient.size() == 3)
                {
                    // Compute the 'active jacobian' at the current location
                    /// We use the RobotState::getJacobian(group, link, point, jacobian)
                    /// function here, which computes the jacobian wrt the given point
                    /// on the given link of the given group. This is equivalent to the
                    /// OpenRAVE function CalculateActiveJacobian
                    //ROS_INFO("Computing 'active jacobian' for link: %s", current_link_name.c_str());
                    Eigen::MatrixXd new_active_jacobian; /// <- Make matrix storage for the gradient
                    bool success = baxter_kinematic_state_->getJacobian(baxter_right_arm_group_.get(), baxter_kinematic_state_->getLinkModel(current_link_name), right_link_points_[link_idx][point_idx], new_active_jacobian); /// <- Do magic here
                    if (!success)
                    {
                        ROS_ERROR("Computing 'active jacobian' for right arm failed");
                    }
                    else
                    {
                        //ROS_INFO("Computed 'active jacobian' for the right arm as a %ld by %ld matrix", new_active_jacobian.rows(), new_active_jacobian.cols());
                    }
                    // Get the translation-only section of the jacobian (3x7)
                    Eigen::Matrix<double, 3, JOINTS> translation_jacobian;
                    translation_jacobian << new_active_jacobian.row(0),new_active_jacobian.row(1),new_active_jacobian.row(2);
                    // Print the matrix to the screen
                    //std::cout << translation_jacobian << std::endl;
                    // Make a new, larger, matrix to store the extended jacobians
                    Eigen::Matrix<double, Eigen::Dynamic, JOINTS> extended_arm_jacobians;
                    extended_arm_jacobians.resize(arm_jacobians.rows() + 3, Eigen::NoChange);
                    // Append the active jacobian to the matrix of jacobians
                    extended_arm_jacobians << arm_jacobians,translation_jacobian;
                    arm_jacobians = extended_arm_jacobians;
                    // Make current gradient into a vector
                    Eigen::Vector3d new_arm_gradient(point_gradient[0], point_gradient[1], point_gradient[2]);
                    // Make a new, larger, matrix to store the extended gradients
                    Eigen::Matrix<double, Eigen::Dynamic, 1> extended_arm_gradients;
                    extended_arm_gradients.resize(arm_gradients.rows() + 3, Eigen::NoChange);
                    // Append the gradient to the vector of gradients
                    extended_arm_gradients << arm_gradients,new_arm_gradient;
                    arm_gradients = extended_arm_gradients;
                    // Blacklist the gradient
                    Eigen::Vector3d cur_gradient(point_gradient[0], point_gradient[1], point_gradient[2]);
                    InsertPenaltyVoxel(sdf_, real_location, cur_gradient, writable_sdf);
                }
            }
        }
        // Check if either matrix is empty, if so, return zero vector
        if (arm_jacobians.rows() == 0 || arm_jacobians.cols() == 0)
        {
            ROS_WARN("Arm jacobians matrix is empty, returning zero gradient");
            std::vector<double> joint_space_cost_gradient{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            return std::pair<std::vector<double>, sdf_tools::SignedDistanceField>(joint_space_cost_gradient, writable_sdf);
        }
        if (arm_gradients.rows() == 0 || arm_gradients.cols() == 0)
        {
            ROS_WARN("Arm gradients matrix is empty, returning zero gradient");
            std::vector<double> joint_space_cost_gradient{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            return std::pair<std::vector<double>, sdf_tools::SignedDistanceField>(joint_space_cost_gradient, writable_sdf);
        }
        //ROS_INFO("Doing Jacobian magic");
        // Do the math
        Eigen::Matrix<double, 7, 1> raw_joint_space_cost_gradient = ProcessJacobian(arm_jacobians) * arm_gradients;
        double inv_norm = 1.0 / raw_joint_space_cost_gradient.norm();
        if (fabs(inv_norm) < RCOND)
        {
            inv_norm = 1.0;
        }
        // Store the computed gradient
        //ROS_INFO("Extracting the joint-space cost gradient");
        std::vector<double> joint_space_cost_gradient(7);
        joint_space_cost_gradient[0] = gradient_multiplier * raw_joint_space_cost_gradient(0,0) * inv_norm;
        joint_space_cost_gradient[1] = gradient_multiplier * raw_joint_space_cost_gradient(1,0) * inv_norm;
        joint_space_cost_gradient[2] = gradient_multiplier * raw_joint_space_cost_gradient(2,0) * inv_norm;
        joint_space_cost_gradient[3] = gradient_multiplier * raw_joint_space_cost_gradient(3,0) * inv_norm;
        joint_space_cost_gradient[4] = gradient_multiplier * raw_joint_space_cost_gradient(4,0) * inv_norm;
        joint_space_cost_gradient[5] = gradient_multiplier * raw_joint_space_cost_gradient(5,0) * inv_norm;
        joint_space_cost_gradient[6] = gradient_multiplier * raw_joint_space_cost_gradient(6,0) * inv_norm;
        return std::pair<std::vector<double>, sdf_tools::SignedDistanceField>(joint_space_cost_gradient, writable_sdf);
    }

    Eigen::Matrix<double, JOINTS, Eigen::Dynamic> ProcessJacobian(Eigen::Matrix<double, Eigen::Dynamic, JOINTS>& jacobian)
    {
#ifdef USE_JACOBIAN_PSEUDOINVERSE
        return EIGEN_PINV::pinv(jacobian, RCOND);
#else
        return jacobian.transpose();
#endif
    }

    double ComputeExpectedLeftArmCost(std::vector<double>& current_configuration, std::vector<double>& control_input, u_int32_t samples)
    {
        //ROS_INFO("Computing estimated control input cost with %u samples", samples);
        double total_cost = 0.0;
        for (u_int32_t idx = 0; idx < samples; idx++)
        {
            //ROS_INFO("Sampling a configuration from the current configuration and control input");
            std::vector<double> real_configuration = model_.SampleLeftArm(current_configuration, control_input);
            //ROS_INFO("Computing transforms from the sampled configuration");
            total_cost += ComputeLeftArmConfigurationCost(real_configuration);
        }
        return total_cost;
    }

    double ComputeExpectedRightArmCost(std::vector<double>& current_configuration, std::vector<double>& control_input, u_int32_t samples)
    {
        //ROS_INFO("Computing estimated control input cost with %u samples", samples);
        double total_cost = 0.0;
        for (u_int32_t idx = 0; idx < samples; idx++)
        {
            //ROS_INFO("Sampling a configuration from the current configuration and control input");
            std::vector<double> real_configuration = model_.SampleRightArm(current_configuration, control_input);
            //ROS_INFO("Computing transforms from the sampled configuration");
            total_cost += ComputeRightArmConfigurationCost(real_configuration);
        }
        return total_cost;
    }

    double ComputeLeftArmConfigurationCost(std::vector<double>& current_configuration)
    {
        double configuration_cost = 0.0;
        std::vector<Eigen::Affine3d> transforms = ComputeLeftArmLinkTransforms(current_configuration);
        for (size_t link_idx = 0; link_idx < JOINTS; link_idx++)
        {
            for (size_t point_idx = 0; point_idx < left_link_points_[link_idx].size(); point_idx++)
            {
                Eigen::Vector3d real_location = transforms[link_idx] * left_link_points_[link_idx][point_idx];
                configuration_cost += LookupInSDF(real_location, sdf_);
            }
        }
        return configuration_cost;
    }

    double ComputeRightArmConfigurationCost(std::vector<double>& current_configuration)
    {
        double configuration_cost = 0.0;
        std::vector<Eigen::Affine3d> transforms = ComputeRightArmLinkTransforms(current_configuration);
        for (size_t link_idx = 0; link_idx < JOINTS; link_idx++)
        {
            for (size_t point_idx = 0; point_idx < right_link_points_[link_idx].size(); point_idx++)
            {
                Eigen::Vector3d real_location = transforms[link_idx] * right_link_points_[link_idx][point_idx];
                configuration_cost += LookupInSDF(real_location, sdf_);
            }
        }
        return configuration_cost;
    }

    double ComputeLeftArmBrittleness(std::vector<double>& current_configuration, double gradient_multiplier)
    {
        // Compute "brittleness" of a configuration
        // Compute the cost of the current configuration
        double current_cost = ComputeLeftArmConfigurationCost(current_configuration);
        // Compute the gradient at the current configuration
        std::vector<double> current_gradient = ComputeAndBlacklistLeftArmCostGradient(current_configuration, sdf_, gradient_multiplier).first;
        // Compute "worst case" configuration which is current-gradient
        std::vector<double> worst_case_configuration(JOINTS);
        for (size_t idx = 0; idx < JOINTS; idx++)
        {
            worst_case_configuration[idx] = current_configuration[idx] - current_gradient[idx];
        }
        // Compute the cost of the "worst case" configuration
        double worst_case_cost = ComputeLeftArmConfigurationCost(worst_case_configuration);
        // Return "brittleness", the difference between current and worst case costs
        return current_cost - worst_case_cost;
    }

    double ComputeRightArmBrittleness(std::vector<double>& current_configuration, double gradient_multiplier)
    {
        // Compute "brittleness" of a configuration
        // Compute the cost of the current configuration
        double current_cost = ComputeRightArmConfigurationCost(current_configuration);
        // Compute the gradient at the current configuration
        std::vector<double> current_gradient = ComputeAndBlacklistRightArmCostGradient(current_configuration, sdf_, gradient_multiplier).first;
        // Compute "worst case" configuration which is current-gradient
        std::vector<double> worst_case_configuration(JOINTS);
        for (size_t idx = 0; idx < JOINTS; idx++)
        {
            worst_case_configuration[idx] = current_configuration[idx] - current_gradient[idx];
        }
        // Compute the cost of the "worst case" configuration
        double worst_case_cost = ComputeRightArmConfigurationCost(worst_case_configuration);
        // Return "brittleness", the difference between current and worst case costs
        return current_cost - worst_case_cost;
    }

    bool ComputeFeatureServiceCB(baxter_uncertainty::ComputeFeatures::Request& req, baxter_uncertainty::ComputeFeatures::Response& res)
    {
        ROS_INFO("Processing new ComputeFeature request...");
        // Sanity check the request
        bool valid_request = IsComputeFeatureRequestValid(req);
        if (valid_request)
        {
            if (req.ArmOption == baxter_uncertainty::ComputeFeaturesRequest::LEFT_ARM_ONLY)
            {
                if (req.FeatureOption == baxter_uncertainty::ComputeFeaturesRequest::COST_ONLY)
                {
                    res.LeftArmCost = ComputeLeftArmConfigurationCost(req.LeftArmConfiguration);
                    std::cout << "Computed configuration cost " << res.LeftArmCost << " for configuration " << PrettyPrint(req.LeftArmConfiguration) << std::endl;
#ifdef ENABLE_DEBUG_DISPLAY_MARKERS
                    point_display_pub_.publish(DrawLeftArmLinksAndLinkPoints(req.LeftArmConfiguration));
#endif
#ifdef ENABLE_VERBOSE_FEATURES
                    std::cout << "Computed configuration cost " << res.LeftArmCost << " for configuration " << PrettyPrint(req.LeftArmConfiguration) << std::endl;
#endif
#ifdef ENABLE_INTERACTIVE_FEATURES
                    std::cout << "Press ENTER to continue..." << std::endl;
                    getchar();
#endif
                }
                else if (req.FeatureOption == baxter_uncertainty::ComputeFeaturesRequest::BRITTLENESS_ONLY)
                {
                    res.LeftArmBrittleness = ComputeLeftArmBrittleness(req.LeftArmConfiguration, req.GradientMultiplier);
#ifdef ENABLE_DEBUG_DISPLAY_MARKERS
                    point_display_pub_.publish(DrawLeftArmPointGradients(req.LeftArmConfiguration, sdf_));
#endif
#ifdef ENABLE_VERBOSE_FEATURES
                    std::cout << "Computed configuration brittleness " << res.LeftArmBrittleness << " for configuration " << PrettyPrint(req.LeftArmConfiguration) << std::endl;
#endif
#ifdef ENABLE_INTERACTIVE_FEATURES
                    std::cout << "Press ENTER to continue..." << std::endl;
                    getchar();
#endif
                }
                else if (req.FeatureOption == baxter_uncertainty::ComputeFeaturesRequest::COST_AND_BRITTLENESS)
                {
                    res.LeftArmCost = ComputeLeftArmConfigurationCost(req.LeftArmConfiguration);
                    res.LeftArmBrittleness = ComputeLeftArmBrittleness(req.LeftArmConfiguration, req.GradientMultiplier);
#ifdef ENABLE_DEBUG_DISPLAY_MARKERS
                    point_display_pub_.publish(DrawLeftArmLinksAndLinkPoints(req.LeftArmConfiguration));
                    point_display_pub_.publish(DrawLeftArmPointGradients(req.LeftArmConfiguration, sdf_));
#endif
#ifdef ENABLE_VERBOSE_FEATURES
                    std::cout << "Computed configuration cost " << res.LeftArmCost << " and brittleness " << res.LeftArmBrittleness << " for configuration " << PrettyPrint(req.LeftArmConfiguration) << std::endl;
#endif
#ifdef ENABLE_INTERACTIVE_FEATURES
                    std::cout << "Press ENTER to continue..." << std::endl;
                    getchar();
#endif
                }
                ROS_INFO("...Left arm features computed");
                res.Status = baxter_uncertainty::ComputeFeaturesResponse::SUCCESS;
            }
            else if (req.ArmOption == baxter_uncertainty::ComputeFeaturesRequest::RIGHT_ARM_ONLY)
            {
                if (req.FeatureOption == baxter_uncertainty::ComputeFeaturesRequest::COST_ONLY)
                {
                    res.RightArmCost = ComputeRightArmConfigurationCost(req.RightArmConfiguration);
#ifdef ENABLE_DEBUG_DISPLAY_MARKERS
                    point_display_pub_.publish(DrawRightArmLinksAndLinkPoints(req.RightArmConfiguration));
#endif
#ifdef ENABLE_VERBOSE_FEATURES
                    std::cout << "Computed configuration cost " << res.RightArmCost << " for configuration " << PrettyPrint(req.RightArmConfiguration) << std::endl;
#endif
#ifdef ENABLE_INTERACTIVE_FEATURES
                    std::cout << "Press ENTER to continue..." << std::endl;
                    getchar();
#endif
                }
                else if (req.FeatureOption == baxter_uncertainty::ComputeFeaturesRequest::BRITTLENESS_ONLY)
                {
                    res.RightArmBrittleness = ComputeRightArmBrittleness(req.RightArmConfiguration, req.GradientMultiplier);
#ifdef ENABLE_DEBUG_DISPLAY_MARKERS
                    point_display_pub_.publish(DrawRightArmPointGradients(req.RightArmConfiguration, sdf_));
#endif
#ifdef ENABLE_VERBOSE_FEATURES
                    std::cout << "Computed configuration brittleness " << res.RightArmBrittleness << " for configuration " << PrettyPrint(req.RightArmConfiguration) << std::endl;
#endif
#ifdef ENABLE_INTERACTIVE_FEATURES
                    std::cout << "Press ENTER to continue..." << std::endl;
                    getchar();
#endif
                }
                else if (req.FeatureOption == baxter_uncertainty::ComputeFeaturesRequest::COST_AND_BRITTLENESS)
                {
                    res.RightArmCost = ComputeRightArmConfigurationCost(req.RightArmConfiguration);
                    res.RightArmBrittleness = ComputeRightArmBrittleness(req.RightArmConfiguration, req.GradientMultiplier);
#ifdef ENABLE_DEBUG_DISPLAY_MARKERS
                    point_display_pub_.publish(DrawRightArmLinksAndLinkPoints(req.RightArmConfiguration));
                    point_display_pub_.publish(DrawRightArmPointGradients(req.RightArmConfiguration, sdf_));
#endif
#ifdef ENABLE_VERBOSE_FEATURES
                    std::cout << "Computed configuration cost " << res.RightArmCost << " and brittleness " << res.RightArmBrittleness << " for configuration " << PrettyPrint(req.RightArmConfiguration) << std::endl;
#endif
#ifdef ENABLE_INTERACTIVE_FEATURES
                    std::cout << "Press ENTER to continue..." << std::endl;
                    getchar();
#endif
                }
                ROS_INFO("...Right arm features computed");
                res.Status = baxter_uncertainty::ComputeFeaturesResponse::SUCCESS;
            }
            else if (req.ArmOption == baxter_uncertainty::ComputeFeaturesRequest::BOTH_ARMS)
            {
                if (req.FeatureOption == baxter_uncertainty::ComputeFeaturesRequest::COST_ONLY)
                {
                    res.LeftArmCost = ComputeLeftArmConfigurationCost(req.LeftArmConfiguration);
                    res.RightArmCost = ComputeRightArmConfigurationCost(req.RightArmConfiguration);
#ifdef ENABLE_DEBUG_DISPLAY_MARKERS
                    point_display_pub_.publish(DrawLinksAndLinkPoints(req.LeftArmConfiguration, req.RightArmConfiguration));
#endif
#ifdef ENABLE_VERBOSE_FEATURES
                    std::cout << "Computed configuration cost " << res.LeftArmCost << " (left) " << res.RightArmCost << " (right) for configuration " << PrettyPrint(req.LeftArmConfiguration) << " (left) " << PrettyPrint(req.RightArmConfiguration) << std::endl;
#endif
#ifdef ENABLE_INTERACTIVE_FEATURES
                    std::cout << "Press ENTER to continue..." << std::endl;
                    getchar();
#endif
                }
                else if (req.FeatureOption == baxter_uncertainty::ComputeFeaturesRequest::BRITTLENESS_ONLY)
                {
                    res.LeftArmBrittleness = ComputeLeftArmBrittleness(req.LeftArmConfiguration, req.GradientMultiplier);
                    res.RightArmBrittleness = ComputeRightArmBrittleness(req.RightArmConfiguration, req.GradientMultiplier);
#ifdef ENABLE_DEBUG_DISPLAY_MARKERS
                    point_display_pub_.publish(DrawPointGradients(req.LeftArmConfiguration, req.RightArmConfiguration, sdf_));
#endif
#ifdef ENABLE_VERBOSE_FEATURES
                    std::cout << "Computed configuration brittleness " << res.LeftArmBrittleness << " (left) " << res.RightArmBrittleness << " (right) for configuration " << PrettyPrint(req.LeftArmConfiguration) << " (left) " << PrettyPrint(req.RightArmConfiguration) << " (right)" << std::endl;
#endif
#ifdef ENABLE_INTERACTIVE_FEATURES
                    std::cout << "Press ENTER to continue..." << std::endl;
                    getchar();
#endif
                }
                else if (req.FeatureOption == baxter_uncertainty::ComputeFeaturesRequest::COST_AND_BRITTLENESS)
                {
                    res.LeftArmCost = ComputeLeftArmConfigurationCost(req.LeftArmConfiguration);
                    res.RightArmCost = ComputeRightArmConfigurationCost(req.RightArmConfiguration);
                    res.LeftArmBrittleness = ComputeLeftArmBrittleness(req.LeftArmConfiguration, req.GradientMultiplier);
                    res.RightArmBrittleness = ComputeRightArmBrittleness(req.RightArmConfiguration, req.GradientMultiplier);
#ifdef ENABLE_DEBUG_DISPLAY_MARKERS
                    point_display_pub_.publish(DrawLinksAndLinkPoints(req.LeftArmConfiguration, req.RightArmConfiguration));
                    point_display_pub_.publish(DrawPointGradients(req.LeftArmConfiguration, req.RightArmConfiguration, sdf_));
#endif
#ifdef ENABLE_VERBOSE_FEATURES
                    std::cout << "Computed configuration cost " << res.LeftArmCost << " (left) " << res.RightArmCost << " (right) and brittleness " << res.LeftArmBrittleness << " (left) " << res.RightArmBrittleness << " (right) for configuration " << PrettyPrint(req.LeftArmConfiguration) << " (left) " << PrettyPrint(req.RightArmConfiguration) << " (right)" << std::endl;
#endif
#ifdef ENABLE_INTERACTIVE_FEATURES
                    std::cout << "Press ENTER to continue..." << std::endl;
                    getchar();
#endif
                }
                ROS_INFO("...Both arm features computed");
                res.Status = baxter_uncertainty::ComputeFeaturesResponse::SUCCESS;
            }
        }
        else
        {
            ROS_ERROR("...Invalid feature computation request");
            res.Status = baxter_uncertainty::ComputeFeaturesResponse::INVALID_REQUEST;
        }
        return true;
    }

    bool IsComputeFeatureRequestValid(baxter_uncertainty::ComputeFeaturesRequest& req)
    {
        // Check the features requested
        if (req.FeatureOption != baxter_uncertainty::ComputeFeaturesRequest::COST_ONLY && req.FeatureOption != baxter_uncertainty::ComputeFeaturesRequest::BRITTLENESS_ONLY && req.FeatureOption != baxter_uncertainty::ComputeFeaturesRequest::COST_AND_BRITTLENESS)
        {
            ROS_ERROR("Invalid features requested");
            return false;
        }
        if (req.GradientMultiplier <= 0.0)
        {
            ROS_ERROR("Invalid GradientMultiplier (<= 0.0 provided)");
            return false;
        }
        else
        {
            ROS_INFO("Running with GradientMultiplier = %f", req.GradientMultiplier);
        }
        // Check the provided arm configurations
        if (req.ArmOption == baxter_uncertainty::ComputeFeaturesRequest::LEFT_ARM_ONLY)
        {
            if (req.LeftArmConfiguration.size() != 7)
            {
                ROS_ERROR("Left arm feature request, but invalid configuration provided");
                return false;
            }
        }
        else if (req.ArmOption == baxter_uncertainty::ComputeFeaturesRequest::RIGHT_ARM_ONLY)
        {
            if (req.RightArmConfiguration.size() != 7)
            {
                ROS_ERROR("Right arm feature request, but invalid configuration provided");
                return false;
            }
        }
        else if (req.ArmOption == baxter_uncertainty::ComputeFeaturesRequest::BOTH_ARMS)
        {
            if (req.LeftArmConfiguration.size() != 7)
            {
                ROS_ERROR("Left arm feature request, but invalid configuration provided");
                return false;
            }
            if (req.RightArmConfiguration.size() != 7)
            {
                ROS_ERROR("Right arm feature request, but invalid configuration provided");
                return false;
            }
        }
        else
        {
            ROS_ERROR("Invalid arm option for feature request");
            return false;
        }
        // If everything has been ok
        return true;
    }

    bool ComputeGradientServiceCB(baxter_uncertainty::ComputeGradientRequest& req, baxter_uncertainty::ComputeGradientResponse& res)
    {
        clock_t st, et;
        st = std::clock();
        ROS_INFO("Processing new ComputeGradient request...");
        // Sanity check the request
        bool valid_request = IsComputeGradientRequestValid(req);
        if (valid_request)
        {
            CONTROL_GENERATION_MODES control_generation_mode = GetControlGenerationMode(req.ControlGenerationMode);
            if (req.ArmGradientOption == baxter_uncertainty::ComputeGradientRequest::LEFT_ARM_ONLY)
            {
                res.LeftArmGradient = EstimateBestLeftArmControlInput(req.LeftArmConfiguration, control_generation_mode, req.MaxControlsToCheck, req.ExpectedCostSamples, req.GradientMultiplier, true);
                ROS_INFO("...Gradient computed for left arm");
                res.Status = baxter_uncertainty::ComputeGradientResponse::SUCCESS;
#ifdef ENABLE_DEBUG_DISPLAY_MARKERS
                // Draw left arm points and gradients
                //point_display_pub_.publish(DrawLeftArmLinksAndLinkPoints(req.LeftArmConfiguration));
                //point_display_pub_.publish(DrawLeftArmPointGradients(req.LeftArmConfiguration, sdf_));
                point_display_pub_.publish(DrawLeftArmControlInput(req.LeftArmConfiguration, res.LeftArmGradient));
#endif
#ifdef ENABLE_INTERACTIVE_CONTROL_INPUTS
                std::cout << "Left arm control input: " << PrettyPrint(res.LeftArmGradient) << std::endl;
                std::cout << "Press ENTER to return computed control input..." << std::endl;
                getchar();
#endif
            }
            else if (req.ArmGradientOption == baxter_uncertainty::ComputeGradientRequest::RIGHT_ARM_ONLY)
            {
                res.RightArmGradient = EstimateBestRightArmControlInput(req.RightArmConfiguration, control_generation_mode, req.MaxControlsToCheck, req.ExpectedCostSamples, req.GradientMultiplier, true);
                ROS_INFO("...Gradient computed for right arm");
                res.Status = baxter_uncertainty::ComputeGradientResponse::SUCCESS;
#ifdef ENABLE_DEBUG_DISPLAY_MARKERS
                // Draw right arm points and gradients
                //point_display_pub_.publish(DrawRightArmLinksAndLinkPoints(req.RightArmConfiguration));
                //point_display_pub_.publish(DrawRightArmPointGradients(req.RightArmConfiguration, sdf_));
                point_display_pub_.publish(DrawRightArmControlInput(req.RightArmConfiguration, res.RightArmGradient));
#endif
#ifdef ENABLE_INTERACTIVE_CONTROL_INPUTS
                std::cout << "Right arm control input: " << PrettyPrint(res.RightArmGradient) << std::endl;
                std::cout << "Press ENTER to return computed control input..." << std::endl;
                getchar();
#endif
            }
            else if (req.ArmGradientOption == baxter_uncertainty::ComputeGradientRequest::BOTH_ARMS)
            {
                res.LeftArmGradient = EstimateBestLeftArmControlInput(req.LeftArmConfiguration, control_generation_mode, req.MaxControlsToCheck, req.ExpectedCostSamples, req.GradientMultiplier, true);
                res.RightArmGradient = EstimateBestRightArmControlInput(req.RightArmConfiguration, control_generation_mode, req.MaxControlsToCheck, req.ExpectedCostSamples, req.GradientMultiplier, false);
                ROS_INFO("...Gradient computed for both arms");
                res.Status = baxter_uncertainty::ComputeGradientResponse::SUCCESS;
#ifdef ENABLE_DEBUG_DISPLAY_MARKERS
                // Draw points and gradients for both arms
                //point_display_pub_.publish(DrawLinksAndLinkPoints(req.LeftArmConfiguration, req.RightArmConfiguration));
                //point_display_pub_.publish(DrawPointGradients(req.LeftArmConfiguration, req.RightArmConfiguration, sdf_));
                point_display_pub_.publish(DrawControlInput(req.LeftArmConfiguration, req.RightArmConfiguration, res.LeftArmGradient, res.RightArmGradient));
#endif
#ifdef ENABLE_INTERACTIVE_CONTROL_INPUTS
                std::cout << "Left arm control input: " << PrettyPrint(res.LeftArmGradient) << std::endl;
                std::cout << "Right arm control input: " << PrettyPrint(res.RightArmGradient) << std::endl;
                std::cout << "Press ENTER to return computed control input..." << std::endl;
                getchar();
#endif
            }
        }
        else
        {
            ROS_ERROR("...Invalid gradient computation request");
            res.Status = baxter_uncertainty::ComputeGradientResponse::INVALID_REQUEST;
        }
        et = std::clock();
        ROS_INFO("Generating control input took %f seconds", (((float)(et - st)) / CLOCKS_PER_SEC));
        return true;
    }

    bool IsComputeGradientRequestValid(baxter_uncertainty::ComputeGradientRequest& req)
    {
        // Check the basic options
        if (GetControlGenerationMode(req.ControlGenerationMode) == UNKNOWNMODE)
        {
            ROS_ERROR("Invalid control generation mode");
            return false;
        }
        if (req.MaxControlsToCheck == 0)
        {
            ROS_ERROR("Invalid number of control options (zero requested)");
            return false;
        }
        if (req.ControlGenerationMode != baxter_uncertainty::ComputeGradientRequest::SAMPLED_EIGHT_CONNECTED && req.ControlGenerationMode != baxter_uncertainty::ComputeGradientRequest::GRADIENT_BLACKLIST)
        {
            ROS_INFO("MaxControlsToCheck will be ignored since the requested control generation mode produces a fixed number of control options");
        }
        else
        {
            ROS_INFO("Running with MaxControlsToCheck = %d", req.MaxControlsToCheck);
        }
        if (req.ExpectedCostSamples == 0)
        {
            ROS_ERROR("Invalid number of cost samples (zero requested)");
            return false;
        }
        else
        {
            ROS_INFO("Running with ExpectedCostSamples = %d", req.ExpectedCostSamples);
        }
        if (req.GradientMultiplier <= 0.0)
        {
            ROS_ERROR("Invalid GradientMultiplier (<= 0.0 provided)");
            return false;
        }
        else
        {
            ROS_INFO("Running with GradientMultiplier = %f", req.GradientMultiplier);
        }
        // Check the provided arm configurations
        if (req.ArmGradientOption == baxter_uncertainty::ComputeGradientRequest::LEFT_ARM_ONLY)
        {
            if (req.LeftArmConfiguration.size() != 7)
            {
                ROS_ERROR("Left arm gradient request, but invalid configuration provided");
                return false;
            }
        }
        else if (req.ArmGradientOption == baxter_uncertainty::ComputeGradientRequest::RIGHT_ARM_ONLY)
        {
            if (req.RightArmConfiguration.size() != 7)
            {
                ROS_ERROR("Right arm gradient request, but invalid configuration provided");
                return false;
            }
        }
        else if (req.ArmGradientOption == baxter_uncertainty::ComputeGradientRequest::BOTH_ARMS)
        {
            if (req.LeftArmConfiguration.size() != 7)
            {
                ROS_ERROR("Left arm gradient request, but invalid configuration provided");
                return false;
            }
            if (req.RightArmConfiguration.size() != 7)
            {
                ROS_ERROR("Right arm gradient request, but invalid configuration provided");
                return false;
            }
        }
        else
        {
            ROS_ERROR("Invalid arm option for gradient request");
            return false;
        }
        // If everything has been ok
        return true;
    }

    CONTROL_GENERATION_MODES GetControlGenerationMode(u_int8_t mode)
    {
        if (mode == baxter_uncertainty::ComputeGradientRequest::FOUR_CONNECTED)
        {
            return FOURCONNECTED;
        }
        else if (mode == baxter_uncertainty::ComputeGradientRequest::HYBRID_FOUR_CONNECTED)
        {
            return HYBRIDFOURCONNECTED;
        }
        else if (mode == baxter_uncertainty::ComputeGradientRequest::SAMPLED_EIGHT_CONNECTED)
        {
            return SAMPLEDEIGHTCONNECTED;
        }
        else if (mode == baxter_uncertainty::ComputeGradientRequest::GRADIENT_BLACKLIST)
        {
            return GRADIENTBLACKLIST;
        }
        else if (mode == baxter_uncertainty::ComputeGradientRequest::POINT_NEIGHBORS)
        {
            return POINTNEIGHBORS;
        }
        else
        {
            return UNKNOWNMODE;
        }
    }

    visualization_msgs::Marker DrawLinksAndLinkPoints(std::vector<double>& left_arm_configuration, std::vector<double>& right_arm_configuration)
    {
        // Get the markers for each arm
        visualization_msgs::Marker left_arm_markers = DrawLeftArmLinksAndLinkPoints(left_arm_configuration);
        visualization_msgs::Marker right_arm_markers = DrawRightArmLinksAndLinkPoints(right_arm_configuration);
        // Combine the markerlists together into a single message
        visualization_msgs::Marker both_arm_markers = left_arm_markers;
        // Insert the right arm markers
        both_arm_markers.points.insert(both_arm_markers.points.end(), right_arm_markers.points.begin(), right_arm_markers.points.end());
        both_arm_markers.colors.insert(both_arm_markers.colors.end(), right_arm_markers.colors.begin(), right_arm_markers.colors.end());
        return both_arm_markers;
    }

    visualization_msgs::Marker DrawLeftArmLinksAndLinkPoints(std::vector<double>& left_arm_configuration)
    {
        visualization_msgs::Marker display_rep;
        // Populate basic operations
        display_rep.header.frame_id = "base";
        display_rep.ns = "point_display";
        display_rep.id = 1;
        display_rep.type = visualization_msgs::Marker::CUBE_LIST;
        display_rep.action = visualization_msgs::Marker::ADD;
        display_rep.lifetime = ros::Duration(0.0);
        display_rep.frame_locked = false;
        display_rep.scale.x = 0.01;
        display_rep.scale.y = 0.01;
        display_rep.scale.z = 0.01;
        // Get the transforms for the left arms
        std::vector<Eigen::Affine3d> left_arm_transforms = ComputeLeftArmLinkTransforms(left_arm_configuration);
        for (size_t link_index = 0; link_index < JOINTS; link_index++)
        {
            // Add the marker for the current left arm link
            geometry_msgs::Point left_link_point;
            std_msgs::ColorRGBA left_link_color;
            left_link_point.x = left_arm_transforms[link_index].translation().x();
            left_link_point.y = left_arm_transforms[link_index].translation().y();
            left_link_point.z = left_arm_transforms[link_index].translation().z();
            left_link_color.a = 1.0;
            left_link_color.r = 1.0;
            left_link_color.g = 1.0;
            left_link_color.b = 0.0;
            display_rep.points.push_back(left_link_point);
            display_rep.colors.push_back(left_link_color);
            // Add markers for each point of the current left arm link
            for (size_t point_index = 0; point_index < left_link_points_[link_index].size(); point_index++)
            {
                Eigen::Vector3d point_location = left_arm_transforms[link_index] * left_link_points_[link_index][point_index];
                geometry_msgs::Point left_link_point_point;
                std_msgs::ColorRGBA left_link_point_color;
                left_link_point_point.x = point_location.x();
                left_link_point_point.y = point_location.y();
                left_link_point_point.z = point_location.z();
                left_link_point_color.a = 1.0;
                left_link_point_color.r = 1.0;
                left_link_point_color.g = 0.0;
                left_link_point_color.b = 0.0;
                display_rep.points.push_back(left_link_point_point);
                display_rep.colors.push_back(left_link_point_color);
            }
        }
        return display_rep;
    }

    visualization_msgs::Marker DrawRightArmLinksAndLinkPoints(std::vector<double>& right_arm_configuration)
    {
        visualization_msgs::Marker display_rep;
        // Populate basic operations
        display_rep.header.frame_id = "base";
        display_rep.ns = "point_display";
        display_rep.id = 1;
        display_rep.type = visualization_msgs::Marker::CUBE_LIST;
        display_rep.action = visualization_msgs::Marker::ADD;
        display_rep.lifetime = ros::Duration(0.0);
        display_rep.frame_locked = false;
        display_rep.scale.x = 0.01;
        display_rep.scale.y = 0.01;
        display_rep.scale.z = 0.01;
        // Get the transforms for the right arm
        std::vector<Eigen::Affine3d> right_arm_transforms = ComputeRightArmLinkTransforms(right_arm_configuration);
        for (size_t link_index = 0; link_index < JOINTS; link_index++)
        {
            // Add the marker for the current right arm link
            geometry_msgs::Point right_link_point;
            std_msgs::ColorRGBA right_link_color;
            right_link_point.x = right_arm_transforms[link_index].translation().x();
            right_link_point.y = right_arm_transforms[link_index].translation().y();
            right_link_point.z = right_arm_transforms[link_index].translation().z();
            right_link_color.a = 1.0;
            right_link_color.r = 0.0;
            right_link_color.g = 1.0;
            right_link_color.b = 0.0;
            display_rep.points.push_back(right_link_point);
            display_rep.colors.push_back(right_link_color);
            // Add markers for each point of the current right arm link
            for (size_t point_index = 0; point_index < right_link_points_[link_index].size(); point_index++)
            {
                Eigen::Vector3d point_location = right_arm_transforms[link_index] * right_link_points_[link_index][point_index];
                geometry_msgs::Point right_link_point_point;
                std_msgs::ColorRGBA right_link_point_color;
                right_link_point_point.x = point_location.x();
                right_link_point_point.y = point_location.y();
                right_link_point_point.z = point_location.z();
                right_link_point_color.a = 1.0;
                right_link_point_color.r = 0.0;
                right_link_point_color.g = 0.0;
                right_link_point_color.b = 1.0;
                display_rep.points.push_back(right_link_point_point);
                display_rep.colors.push_back(right_link_point_color);
            }
        }
        return display_rep;
    }

    visualization_msgs::Marker DrawControlInput(std::vector<double>& left_arm_configuration, std::vector<double>& right_arm_configuration, std::vector<double>& left_arm_control_input, std::vector<double>& right_arm_control_input)
    {
        // Get the markers for each arm
        visualization_msgs::Marker left_arm_markers = DrawLeftArmControlInput(left_arm_configuration, left_arm_control_input);
        visualization_msgs::Marker right_arm_markers = DrawRightArmControlInput(right_arm_configuration, right_arm_control_input);
        // Combine the markerlists together into a single message
        visualization_msgs::Marker both_arm_markers = left_arm_markers;
        // Insert the right arm markers
        both_arm_markers.points.insert(both_arm_markers.points.end(), right_arm_markers.points.begin(), right_arm_markers.points.end());
        both_arm_markers.colors.insert(both_arm_markers.colors.end(), right_arm_markers.colors.begin(), right_arm_markers.colors.end());
        return both_arm_markers;
    }

    visualization_msgs::Marker DrawLeftArmControlInput(std::vector<double>& left_arm_configuration, std::vector<double>& left_arm_control_input)
    {
        // Check the input sizes
        if (left_arm_configuration.size() != 7)
        {
            left_arm_configuration = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            ROS_WARN("Left arm configuration is invalid, setting to zero");
        }
        if (left_arm_control_input.size() != 7)
        {
            ROS_WARN("Left arm control input is invalid (%lu elements), setting to zero", left_arm_control_input.size());
            left_arm_control_input = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        }
        visualization_msgs::Marker display_rep;
        // Populate basic operations
        display_rep.header.frame_id = "base";
        display_rep.ns = "control_display";
        display_rep.id = 1;
        display_rep.type = visualization_msgs::Marker::LINE_LIST;
        display_rep.action = visualization_msgs::Marker::ADD;
        display_rep.lifetime = ros::Duration(0.0);
        display_rep.frame_locked = false;
        display_rep.scale.x = 0.05;
        // Get the transforms for the arm in the current configuration
        std::vector<Eigen::Affine3d> left_arm_transforms = ComputeLeftArmLinkTransforms(left_arm_configuration);
        // Combine the control input with the current configuration
        std::vector<double> left_arm_configuration_with_control_input(JOINTS);
        for (size_t idx = 0; idx < JOINTS; idx++)
        {
            left_arm_configuration_with_control_input[idx] = left_arm_configuration[idx] + left_arm_control_input[idx];
        }
        // Get the transforms for the arm in the current + control input congfiguration
        std::vector<Eigen::Affine3d> left_arm_with_input_transforms = ComputeLeftArmLinkTransforms(left_arm_configuration_with_control_input);
        // Draw a line for each link
        for (size_t link_index = 0; link_index < JOINTS; link_index++)
        {
            // Start transform
            Eigen::Affine3d start_transform = left_arm_transforms[link_index];
            Eigen::Vector3d start_translation = start_transform * baxter_basic_link_offsets_[link_index];
            // Make the start point of the line (the current config)
            geometry_msgs::Point start_point;
            std_msgs::ColorRGBA start_color;
            start_point.x = start_translation.x();
            start_point.y = start_translation.y();
            start_point.z = start_translation.z();
            start_color.a = 1.0;
            start_color.r = 1.0;
            start_color.g = 0.0;
            start_color.b = 0.0;
            // End transform
            Eigen::Affine3d end_transform = left_arm_with_input_transforms[link_index];
            Eigen::Vector3d end_translation = end_transform * baxter_basic_link_offsets_[link_index];
            // Make the end point of the line (the current+input config)
            geometry_msgs::Point end_point;
            std_msgs::ColorRGBA end_color;
            end_point.x = end_translation.x();
            end_point.y = end_translation.y();
            end_point.z = end_translation.z();
            end_color.a = 1.0;
            end_color.r = 1.0;
            end_color.g = 1.0;
            end_color.b = 0.0;
            // Add the points
            display_rep.points.push_back(start_point);
            display_rep.colors.push_back(start_color);
            display_rep.points.push_back(end_point);
            display_rep.colors.push_back(end_color);
        }
        return display_rep;
    }

    visualization_msgs::Marker DrawRightArmControlInput(std::vector<double>& right_arm_configuration, std::vector<double>& right_arm_control_input)
    {
        // Check the input sizes
        if (right_arm_configuration.size() != 7)
        {
            right_arm_configuration = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            ROS_WARN("Left arm configuration is invalid, setting to zero");
        }
        if (right_arm_control_input.size() != 7)
        {
            ROS_WARN("Right arm control input is invalid (%lu elements), setting to zero", right_arm_control_input.size());
            right_arm_control_input = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        }
        visualization_msgs::Marker display_rep;
        // Populate basic operations
        display_rep.header.frame_id = "base";
        display_rep.ns = "control_display";
        display_rep.id = 1;
        display_rep.type = visualization_msgs::Marker::LINE_LIST;
        display_rep.action = visualization_msgs::Marker::ADD;
        display_rep.lifetime = ros::Duration(0.0);
        display_rep.frame_locked = false;
        display_rep.scale.x = 0.05;
        // Get the transforms for the arm in the current configuration
        std::vector<Eigen::Affine3d> right_arm_transforms = ComputeRightArmLinkTransforms(right_arm_configuration);
        // Combine the control input with the current configuration
        std::vector<double> right_arm_configuration_with_control_input(JOINTS);
        for (size_t idx = 0; idx < JOINTS; idx++)
        {
            right_arm_configuration_with_control_input[idx] = right_arm_configuration[idx] + right_arm_control_input[idx];
        }
        // Get the transforms for the arm in the current + control input congfiguration
        std::vector<Eigen::Affine3d> right_arm_with_input_transforms = ComputeRightArmLinkTransforms(right_arm_configuration_with_control_input);
        // Draw a line for each link
        for (size_t link_index = 0; link_index < JOINTS; link_index++)
        {
            // Start transform
            Eigen::Affine3d start_transform = right_arm_transforms[link_index];
            Eigen::Vector3d start_translation = start_transform * baxter_basic_link_offsets_[link_index];
            // Make the start point of the line (the current config)
            geometry_msgs::Point start_point;
            std_msgs::ColorRGBA start_color;
            start_point.x = start_translation.x();
            start_point.y = start_translation.y();
            start_point.z = start_translation.z();
            start_color.a = 1.0;
            start_color.r = 0.0;
            start_color.g = 0.0;
            start_color.b = 1.0;
            // End transform
            Eigen::Affine3d end_transform = right_arm_with_input_transforms[link_index];
            Eigen::Vector3d end_translation = end_transform * baxter_basic_link_offsets_[link_index];
            // Make the end point of the line (the current+input config)
            geometry_msgs::Point end_point;
            std_msgs::ColorRGBA end_color;
            end_point.x = end_translation.x();
            end_point.y = end_translation.y();
            end_point.z = end_translation.z();
            end_color.a = 1.0;
            end_color.r = 0.0;
            end_color.g = 1.0;
            end_color.b = 0.0;
            // Add the points
            display_rep.points.push_back(start_point);
            display_rep.colors.push_back(start_color);
            display_rep.points.push_back(end_point);
            display_rep.colors.push_back(end_color);
        }
        return display_rep;
    }

    visualization_msgs::Marker DrawPointGradients(std::vector<double>& left_arm_configuration, std::vector<double>& right_arm_configuration, sdf_tools::SignedDistanceField& sdf)
    {
        // Get the markers for each arm
        visualization_msgs::Marker left_arm_markers = DrawLeftArmPointGradients(left_arm_configuration, sdf);
        visualization_msgs::Marker right_arm_markers = DrawRightArmPointGradients(right_arm_configuration, sdf);
        // Combine the markerlists together into a single message
        visualization_msgs::Marker both_arm_markers = left_arm_markers;
        // Insert the right arm markers
        both_arm_markers.points.insert(both_arm_markers.points.end(), right_arm_markers.points.begin(), right_arm_markers.points.end());
        both_arm_markers.colors.insert(both_arm_markers.colors.end(), right_arm_markers.colors.begin(), right_arm_markers.colors.end());
        return both_arm_markers;
    }

    visualization_msgs::Marker DrawLeftArmPointGradients(std::vector<double>& left_arm_configuration, sdf_tools::SignedDistanceField& sdf)
    {
        visualization_msgs::Marker display_rep;
        // Populate basic operations
        display_rep.header.frame_id = "base";
        display_rep.ns = "gradient_display";
        display_rep.id = 1;
        display_rep.type = visualization_msgs::Marker::LINE_LIST;
        display_rep.action = visualization_msgs::Marker::ADD;
        display_rep.lifetime = ros::Duration(0.0);
        display_rep.frame_locked = false;
        display_rep.scale.x = 0.01;
        // Get the transforms for the left arms
        std::vector<Eigen::Affine3d> left_arm_transforms = ComputeLeftArmLinkTransforms(left_arm_configuration);
        for (size_t link_index = 0; link_index < JOINTS; link_index++)
        {
            // Add markers for each point of the current left arm link
            for (size_t point_index = 0; point_index < left_link_points_[link_index].size(); point_index++)
            {
                // Get the point location
                Eigen::Vector3d point_location = left_arm_transforms[link_index] * left_link_points_[link_index][point_index];
                // Get the gradient
                std::vector<double> point_gradient = LookupGradientInSDF(point_location, sdf);
                // Check if we have a valid gradient
                if (point_gradient.size() == 3)
                {
                    // If so, we draw a line to show the gradient at the point
                    // Make the start point
                    geometry_msgs::Point start_point;
                    std_msgs::ColorRGBA start_color;
                    start_point.x = point_location.x();
                    start_point.y = point_location.y();
                    start_point.z = point_location.z();
                    start_color.a = 1.0;
                    start_color.r = 1.0;
                    start_color.g = 0.0;
                    start_color.b = 0.0;
                    // Make the end point
                    geometry_msgs::Point end_point;
                    std_msgs::ColorRGBA end_color;
                    end_point.x = point_location.x() + point_gradient[0];
                    end_point.y = point_location.y() + point_gradient[1];
                    end_point.z = point_location.z() + point_gradient[2];
                    end_color.a = 1.0;
                    end_color.r = 1.0;
                    end_color.g = 1.0;
                    end_color.b = 0.0;
                    // Add the points
                    display_rep.points.push_back(start_point);
                    display_rep.colors.push_back(start_color);
                    display_rep.points.push_back(end_point);
                    display_rep.colors.push_back(end_color);
                }
                // If not, we don't draw anything
            }
        }
        return display_rep;
    }

    visualization_msgs::Marker DrawRightArmPointGradients(std::vector<double>& right_arm_configuration, sdf_tools::SignedDistanceField& sdf)
    {
        visualization_msgs::Marker display_rep;
        // Populate basic operations
        display_rep.header.frame_id = "base";
        display_rep.ns = "gradient_display";
        display_rep.id = 1;
        display_rep.type = visualization_msgs::Marker::LINE_LIST;
        display_rep.action = visualization_msgs::Marker::ADD;
        display_rep.lifetime = ros::Duration(0.0);
        display_rep.frame_locked = false;
        display_rep.scale.x = 0.01;
        // Get the transforms for the right arms
        std::vector<Eigen::Affine3d> right_arm_transforms = ComputeRightArmLinkTransforms(right_arm_configuration);
        for (size_t link_index = 0; link_index < JOINTS; link_index++)
        {
            // Add markers for each point of the current left arm link
            for (size_t point_index = 0; point_index < right_link_points_[link_index].size(); point_index++)
            {
                // Get the point location
                Eigen::Vector3d point_location = right_arm_transforms[link_index] * right_link_points_[link_index][point_index];
                // Get the gradient
                std::vector<double> point_gradient = LookupGradientInSDF(point_location, sdf);
                // Check if we have a valid gradient
                if (point_gradient.size() == 3)
                {
                    if (point_gradient[0] == INFINITY || point_gradient[1] == INFINITY || point_gradient[2] == INFINITY)
                    {
                        point_gradient = {0.0, 0.0, 0.0};
                    }
                    else if (isnan(point_gradient[0]) || isnan(point_gradient[1]) || isnan(point_gradient[2]))
                    {
                        point_gradient = {0.0, 0.0, 0.0};
                    }
                    // If so, we draw a line to show the gradient at the point
                    // Make the start point
                    geometry_msgs::Point start_point;
                    std_msgs::ColorRGBA start_color;
                    start_point.x = point_location.x();
                    start_point.y = point_location.y();
                    start_point.z = point_location.z();
                    start_color.a = 1.0;
                    start_color.r = 0.0;
                    start_color.g = 0.0;
                    start_color.b = 1.0;
                    // Make the end point
                    geometry_msgs::Point end_point;
                    std_msgs::ColorRGBA end_color;
                    end_point.x = point_location.x() + point_gradient[0];
                    end_point.y = point_location.y() + point_gradient[1];
                    end_point.z = point_location.z() + point_gradient[2];
                    end_color.a = 1.0;
                    end_color.r = 0.0;
                    end_color.g = 1.0;
                    end_color.b = 0.0;
                    // Add the points
                    display_rep.points.push_back(start_point);
                    display_rep.colors.push_back(start_color);
                    display_rep.points.push_back(end_point);
                    display_rep.colors.push_back(end_color);
                }
                // If not, we don't draw anything
            }
        }
        return display_rep;
    }

    bool ExportInternalSDFToFile(std::string& filepath)
    {
        return sdf_.SaveToFile(filepath);
    }

    bool ImportInternalSDFFromFile(std::string& filepath)
    {
        sdf_tools::SignedDistanceField sdf_copy = sdf_;
        bool loaded = sdf_copy.LoadFromFile(filepath);
        if (loaded)
        {
            sdf_ = sdf_copy;
            ROS_INFO("Loaded internal SDF from file");
            return true;
        }
        else
        {
            ROS_ERROR("Unable to load SDF from file");
            return false;
        }
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "static_baxter_cost_joint_uncertainty_server");
    ROS_INFO("Starting Baxter cost+joint uncertainty computation server [static]...");
    ros::NodeHandle nh;
    ros::NodeHandle nhp("~");
    std::string left_arm_points_dir;
    std::string right_arm_points_dir;
    std::string feature_service;
    std::string gradient_service;
    nhp.param(std::string("left_arm_points_dir"), left_arm_points_dir, std::string("/home/calderpg/Dropbox/ROS_workspace/src/Research/baxter_uncertainty/data/left_arm_points"));
    nhp.param(std::string("right_arm_points_dir"), right_arm_points_dir, std::string("/home/calderpg/Dropbox/ROS_workspace/src/Research/baxter_uncertainty/data/right_arm_points"));
    nhp.param(std::string("feature_service"), feature_service, std::string("compute_baxter_cost_features"));
    nhp.param(std::string("gradient_service"), gradient_service, std::string("compute_baxter_cost_uncertainty_gradient"));
    // Building uncertainty computation
    unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 PRNG(seed);
    std::string filepath("/home/calderpg/Dropbox/ROS_workspace/src/Research/baxter_uncertainty/data/test_scene.sdf");
    StaticBaxterJointUncertaintyCostGradientComputer processor(nh, PRNG, left_arm_points_dir, right_arm_points_dir, filepath);
    // Make service handler
    ros::ServiceServer feature_server = nh.advertiseService(feature_service, &StaticBaxterJointUncertaintyCostGradientComputer::ComputeFeatureServiceCB, &processor);
    ros::ServiceServer gradient_server = nh.advertiseService(gradient_service, &StaticBaxterJointUncertaintyCostGradientComputer::ComputeGradientServiceCB, &processor);
    ROS_INFO("...startup complete");
    ros::Rate spin_rate(100);
    while (ros::ok())
    {
        // Process requests
        ros::spinOnce();
        spin_rate.sleep();
    }
    return 0;
}
