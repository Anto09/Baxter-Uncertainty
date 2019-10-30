#!/usr/bin/python

import math
import os
import sys
from xtf.xtf import *

def process_xtf_file(filename):
    parser = XTFParser()
    xtf_traj = parser.ParseTraj(filename)
    expected_cost_sum = 0.0
    executed_cost_sum = 0.0
    corrected_cost_sum = 0.0
    improvements = []
    for state in xtf_traj.trajectory:
        if "reference_cost" not in state.extras.keys():
            state.extras["reference_cost"] = state.extras["state_cost"]
        expected_cost = state.extras["reference_cost"]
        executed_cost = state.extras["state_cost"]
        corrected_cost = state.extras["executed_cost"]
        improvements.append(corrected_cost - expected_cost)
        expected_cost_sum += expected_cost
        executed_cost_sum += executed_cost
        corrected_cost_sum += corrected_cost
    print("Expected cost: " + str(expected_cost_sum))
    print("Executed cost: " + str(executed_cost_sum))
    print("Corrected cost: " + str(corrected_cost_sum))
    sum_cost_improvement = executed_cost_sum - expected_cost_sum
    print("Sum cost improvement: " + str(sum_cost_improvement))
    avg_improvement = 0.0
    for improvement in improvements:
        avg_improvement += improvement
    avg_improvement = avg_improvement / len(improvements)
    print("Average per-state cost improvement: " + str(avg_improvement))
    return [expected_cost_sum, executed_cost_sum, corrected_cost_sum]

def process_files(directory):
    # Get the xtf files in the current directory
    files = os.listdir(directory)
    xtf_files = []
    for filename in files:
        if ".xtf" in filename:
            xtf_files.append(filename)
    # Process each of the files in turn
    total_expected_cost = 0.0
    total_executed_cost = 0.0
    total_corrected_cost = 0.0
    for filename in xtf_files:
        [expected_cost, executed_cost, corrected_cost] = process_xtf_file(filename)
        total_expected_cost += expected_cost
        total_executed_cost += executed_cost
        total_corrected_cost += corrected_cost
    # Compute the average costs
    average_expected_cost = total_expected_cost / len(xtf_files)
    average_executed_cost = total_executed_cost / len(xtf_files)
    average_corrected_cost = total_corrected_cost / len(xtf_files)
    print("Average expected cost: " + str(average_expected_cost))
    print("Average executed cost: " + str(average_executed_cost))
    print("Average corrected cost: " + str(average_corrected_cost))

if __name__ == '__main__':
    process_files("./")
