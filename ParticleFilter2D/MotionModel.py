import numpy as np

from ParticleFilter2D.Pose2D import Pose2D

class MotionModel:
    """ Probabilistic Motion Model based on example from CS427 Particle Filter Notes.

    Attributes:
        translationStd: standard deviation of translation error used for both x and y
        rotationStd: standard deviation of rotation error
    """

    def __init__(self, translationVar=0.02, rotationVar=0.0174532924):
        """ Initialises MotionModel with default or provided variances

        Args:
            translationVar (float): translation error *variance* i.e. this is equal to
                the stddev^2. (default: 0.02)
            rotationVar (float): rotation error *variance* in radians i.e. this is equal to
                the stddev^2 (default: 0.0174532924 rads (i.e. 1 deg))
        """
        self.translationStd = np.sqrt(translationVar)
        self.rotationStd = np.sqrt(rotationVar)

    def propagatePoses(self, poses, odometry):
        """ Probabilistically applies input odometry to propate the given poses

        Applies the probabilistic motion model from CS427 particle filter notes by propagating
        the set of input poses according to the input odometry, where the motion includes
        the addition of Gaussian noise with the noise parameters stored in the corresponding
        member variables.

        Args:
            poses (np.array): Nx3 numpy array of poses where each pose
                contains the pose's [x, y, theta] values where theta is stored
                in radians.
            odometry (np.array): 1x3 numpy array of odometry parameters 
                [dx, dy, dtheta] 

        Returns:
            np.array: Nx3 numpy array of updated poses where each pose contains the pose's
                [x, y, theta] values where theta is stored in radians. 
        """
        dx, dy, dtheta = odometry

        for i in range(len(poses)):
            translation_noise = np.random.normal(0, self.translationStd, 2)
            rotation_noise = np.random.normal(0, self.rotationStd, 1)
            noise = np.concatenate((translation_noise, rotation_noise))

            # Apply odometry in the robot's local coordinate frame
            local_dx = np.cos(poses[i, 2]) * dx - np.sin(poses[i, 2]) * dy  # Change this line
            local_dy = np.sin(poses[i, 2]) * dx + np.cos(poses[i, 2]) * dy  # Change this line

            poses[i, 0] += local_dx + noise[0]
            poses[i, 1] += local_dy + noise[1]
            poses[i, 2] += dtheta + noise[2]

        return poses