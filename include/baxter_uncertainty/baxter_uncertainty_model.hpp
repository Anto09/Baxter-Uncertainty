#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <random>

#ifndef BAXTER_UNCERTAINTY_MODEL_HPP
#define BAXTER_UNCERTAINTY_MODEL_HPP

namespace BaxterUncertainty
{
    class BaxterJointUncertaintyModel
    {
    protected:

        std::mt19937_64 PRNG_;

    public:

        BaxterJointUncertaintyModel(std::mt19937_64& PRNG) : PRNG_(PRNG) {}

        std::vector<double> SampleLeftArm(std::vector<double>& configuration, std::vector<double>& control_input);

        std::vector<double> SampleRightArm(std::vector<double>& configuration, std::vector<double>& control_input);
    };
}

#endif // BAXTER_UNCERTAINTY_MODEL_HPP
