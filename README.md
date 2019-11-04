# Baxter-Uncertainty
Probabilistic planning with the Baxter robot

This Project was for WPI CS 525: Graduate Motion Planning
It was intended to be a publication worthy work (with a little bit of polishing)

Rethink Baxter was a cheap robot that had problems with exact joint-space control; commanding it the limbs to a certain joint angle resulted in an actual joint angle within some vicinity of the command joint angle.
This uncertainty was first modeled by commanding the robot to a list of given joint angles multiple times, then recording the actual joint angles achieved.
A probability distribution (assumed to be Gaussian) for each commanded joint angle was then built using the achieved joint angles.

The uncertainty balls regions in joint space were then converted to free-space (collision-free) uncertainty spheres which measure the probability that a limb will collide with an object.
Collision detection is performed using a fast signed-distance-field (SDF) checker. The SDF also provides a gradient field in free space which directs the limbs away from each obstacle.

Commanded joint values are associated with the existing Gaussians based on minimum Mahalanobis distance. 
For joint values which are too 'far' from any existing distribution (no previous data) a new distribution was created using an approximation derived from all existing distributions.

Faster collision checking is then performed inside joint space using hyperellipses defined by the distributions' means and covariance matrix eigenvectors.
Joint values inside collision-free hyper ellipse isocontours are then considered to be viable targets.

Future work was to further refine the probability distributions and SDF representation based on repeated execution of motions.

Planning for the robot was performed using OpenRAVE (http://openrave.org/) in C++ with a custom built Baxter model.
Real-time trajectory execution and SDF computation was performed with Moveit! (https://moveit.ros.org/index.html) in C++.
