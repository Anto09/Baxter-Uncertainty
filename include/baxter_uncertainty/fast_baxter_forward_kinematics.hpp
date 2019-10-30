#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <Eigen/Geometry>

#ifndef FAST_BAXTER_FORWARD_KINEMATICS_HPP
#define FAST_BAXTER_FORWARD_KINEMATICS_HPP

namespace FastBaxterForwardKinematics
{
    /*
     *  Compute the 7 homogenous transformation matrices, one for each link of Baxter's left arm
     */
    inline std::vector<Eigen::Transform<double,3,Eigen::Affine> > ComputeLeftArmLinkTransforms(std::vector<double>& joint_values)
    {
        // Make the 4x4 matrix for the first link
        Eigen::Transform<double,3,Eigen::Affine> link1;
        link1.matrix() << cos(joint_values[0]), 0.0, -sin(joint_values[0]), (cos(joint_values[0]) * 0.069),
                sin(joint_values[0]), 0.0, cos(joint_values[0]), (sin(joint_values[0]) * 0.069),
                0.0, -1.0, 0.0, 0.27035,
                0.0, 0.0, 0.0, 1.0;
        // Make the 4x4 matrix for the second link
        Eigen::Transform<double,3,Eigen::Affine> link2;
        link2.matrix() << cos(joint_values[1]), 0.0, sin(joint_values[1]), 0.0,
                sin(joint_values[1]), 0.0, -cos(joint_values[1]), 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0;
        // Make the 4x4 matrix for the third link
        Eigen::Transform<double,3,Eigen::Affine> link3;
        link3.matrix() << cos(joint_values[2]), 0.0, -sin(joint_values[2]), (cos(joint_values[2]) * 0.069),
                sin(joint_values[2]), 0.0, cos(joint_values[2]), (sin(joint_values[2]) * 0.069),
                0.0, -1.0, 0.0, 0.36435,
                0.0, 0.0, 0.0, 1.0;
        // Make the 4x4 matrix for the fourth link
        Eigen::Transform<double,3,Eigen::Affine> link4;
        link4.matrix() << cos(joint_values[3]), 0.0, sin(joint_values[3]), 0.0,
                sin(joint_values[3]), 0.0, -cos(joint_values[3]), 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0;
        // Make the 4x4 matrix for the fifth link
        Eigen::Transform<double,3,Eigen::Affine> link5;
        link5.matrix() << cos(joint_values[4]), 0.0, -sin(joint_values[4]), (cos(joint_values[4] * 0.01)),
                sin(joint_values[4]), 0.0, cos(joint_values[4]), (sin(joint_values[4] * 0.01)),
                0.0, 1.0, 0.0, 0.37429,
                0.0, 0.0, 0.0, 1.0;
        // Make the 4x4 matrix for the sixth link
        Eigen::Transform<double,3,Eigen::Affine> link6;
        link6.matrix() << cos(joint_values[5]), 0.0, sin(joint_values[5]), 0.0,
                sin(joint_values[5]), 0.0, -cos(joint_values[5]), 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0;
        // Make the 4x4 matrix for the seventh link
        Eigen::Transform<double,3,Eigen::Affine> link7;
        link7.matrix() << cos(joint_values[6]), -sin(joint_values[6]), 0.0, 0.0,
                sin(joint_values[6]), cos(joint_values[6]), 0.0, 0.0,
                0.0, 0.0, 1.0, 0.229525,
                0.0, 0.0, 0.0, 1.0;
        // Make the 4x4 matrix for the base offset
        Eigen::Transform<double,3,Eigen::Affine> base_offset;
        base_offset.matrix() << 0.7071, -0.7071, 0.0, 0.0640,
                0.7071, 0.7071, 0.0, 0.2590,
                0.0, 0.0, 1.0, 0.1296,
                0.0, 0.0, 0.0, 1.0;
        ////////////////////////////////////////////////////////////
        // Make the transforms from base to each link
        std::vector<Eigen::Transform<double,3,Eigen::Affine> > link_transforms(7);
        link_transforms[0] = base_offset * link1;
        link_transforms[1] = link_transforms[0] * link2;
        link_transforms[2] = link_transforms[1] * link3;
        link_transforms[3] = link_transforms[2] * link4;
        link_transforms[4] = link_transforms[3] * link5;
        link_transforms[5] = link_transforms[4] * link6;
        link_transforms[6] = link_transforms[5] * link7;
        return link_transforms;
    }

    /*
     *  Compute the 7 homogenous transformation matrices, one for each link of Baxter's right arm
     */
    inline std::vector<Eigen::Transform<double,3,Eigen::Affine> > ComputeRightArmLinkTransforms(std::vector<double>& joint_values)
    {
        // Make the 4x4 matrix for the first link
        Eigen::Transform<double,3,Eigen::Affine> link1;
        link1.matrix() << cos(joint_values[0]), 0.0, -sin(joint_values[0]), (cos(joint_values[0]) * 0.069),
                sin(joint_values[0]), 0.0, cos(joint_values[0]), (sin(joint_values[0]) * 0.069),
                0.0, -1.0, 0.0, 0.27035,
                0.0, 0.0, 0.0, 1.0;
        // Make the 4x4 matrix for the second link
        Eigen::Transform<double,3,Eigen::Affine> link2;
        link2.matrix() << cos(joint_values[1]), 0.0, sin(joint_values[1]), 0.0,
                sin(joint_values[1]), 0.0, -cos(joint_values[1]), 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0;
        // Make the 4x4 matrix for the third link
        Eigen::Transform<double,3,Eigen::Affine> link3;
        link3.matrix() << cos(joint_values[2]), 0.0, -sin(joint_values[2]), (cos(joint_values[2]) * 0.069),
                sin(joint_values[2]), 0.0, cos(joint_values[2]), (sin(joint_values[2]) * 0.069),
                0.0, -1.0, 0.0, 0.36435,
                0.0, 0.0, 0.0, 1.0;
        // Make the 4x4 matrix for the fourth link
        Eigen::Transform<double,3,Eigen::Affine> link4;
        link4.matrix() << cos(joint_values[3]), 0.0, sin(joint_values[3]), 0.0,
                sin(joint_values[3]), 0.0, -cos(joint_values[3]), 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0;
        // Make the 4x4 matrix for the fifth link
        Eigen::Transform<double,3,Eigen::Affine> link5;
        link5.matrix() << cos(joint_values[4]), 0.0, -sin(joint_values[4]), (cos(joint_values[4] * 0.01)),
                sin(joint_values[4]), 0.0, cos(joint_values[4]), (sin(joint_values[4] * 0.01)),
                0.0, 1.0, 0.0, 0.37429,
                0.0, 0.0, 0.0, 1.0;
        // Make the 4x4 matrix for the sixth link
        Eigen::Transform<double,3,Eigen::Affine> link6;
        link6.matrix() << cos(joint_values[5]), 0.0, sin(joint_values[5]), 0.0,
                sin(joint_values[5]), 0.0, -cos(joint_values[5]), 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0;
        // Make the 4x4 matrix for the seventh link
        Eigen::Transform<double,3,Eigen::Affine> link7;
        link7.matrix() << cos(joint_values[6]), -sin(joint_values[6]), 0.0, 0.0,
                sin(joint_values[6]), cos(joint_values[6]), 0.0, 0.0,
                0.0, 0.0, 1.0, 0.229525,
                0.0, 0.0, 0.0, 1.0;
        // Make the 4x4 matrix for the base offset
        Eigen::Transform<double,3,Eigen::Affine> base_offset;
        base_offset.matrix() << 0.7071, 0.7071, 0.0, 0.0640,
                -0.7071, 0.7071, 0.0, -0.2590,
                0.0, 0.0, 1.0, 0.1296,
                0.0, 0.0, 0.0, 1.0;
        ////////////////////////////////////////////////////////////
        // Make the transforms from base to each link
        std::vector<Eigen::Transform<double,3,Eigen::Affine> > link_transforms(7);
        link_transforms[0] = base_offset * link1;
        link_transforms[1] = link_transforms[0] * link2;
        link_transforms[2] = link_transforms[1] * link3;
        link_transforms[3] = link_transforms[2] * link4;
        link_transforms[4] = link_transforms[3] * link5;
        link_transforms[5] = link_transforms[4] * link6;
        link_transforms[6] = link_transforms[5] * link7;
        return link_transforms;
    }
}

#endif // FAST_BAXTER_FORWARD_KINEMATICS_HPP
