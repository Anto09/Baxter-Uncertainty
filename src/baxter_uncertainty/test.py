#!/usr/bin/env python

#   Calder Phillips-Grafflin

import rospy
from baxter_uncertainty.srv import *

def test_gradient_service():
    client = rospy.ServiceProxy("compute_baxter_cost_uncertainty_gradient", ComputeGradient)
    req = ComputeGradientRequest()
    req.ArmGradientOption = ComputeGradientRequest.BOTH_ARMS
    req.ControlGenerationMode = ComputeGradientRequest.FOUR_CONNECTED
    req.MaxControlsToCheck = 100
    req.ExpectedCostSamples = 100
    #req.LeftArmConfiguration = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    req.LeftArmConfiguration = [-0.014189322271728517, -0.04908738515625, -0.757019517956543, 0.004985437554931641, -0.028762139739990235, 0.042951462011718754, -0.05138835633544922]
    #req.RightArmConfiguration = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    req.RightArmConfiguration = [-0.027228158953857422, 0.05867476506958008, 0.8233641869567871, -0.019941750219726564, -0.031446606115722656, -0.05330583231811524, -0.016106798254394532]
    req.GradientMultiplier = 0.1
    res = client.call(req)
    print res

def test_feature_service():
    client = rospy.ServiceProxy("compute_baxter_cost_features", ComputeFeatures)
    req = ComputeFeaturesRequest()
    req.ArmOption = ComputeFeaturesRequest.BOTH_ARMS
    req.FeatureOption = ComputeFeaturesRequest.COST_ONLY
    req.LeftArmConfiguration = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #req.LeftArmConfiguration = [0.2903058637756348, -0.009203884716796876, 0.056373793890380865, 1.672806047277832, -3.0438013748840334, 0.1100631214050293, 3.0514712788146974]
    req.RightArmConfiguration = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #req.RightArmConfiguration = [-0.4030534515563965, 0.16528642970581056, -0.008053399127197266, 1.4473108717163088, 3.0595246779418948, 0.17602429520874024, -2.341238174835205]
    req.GradientMultiplier = 0.2
    res = client.call(req)
    print res


def main():
    #print "Testing features..."
    #test_feature_service()
    print "Testing gradient..."
    test_gradient_service()
    print "Done!"

if __name__ == '__main__':
    main()
