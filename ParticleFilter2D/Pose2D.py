import numpy as np

class Pose2D:
    """Helper class containing a number static methods for creating and manipulating 2D poses"""

    @staticmethod
    def randomPoses(num_poses, worldsize):
        """Generates a numpy array of random poses

        Args:
            num_poses (int): number of poses to generate
            worldsize  (float): specifies dimensions of the world (i.e. the 
                maximum values for the pose x and y values).
        
        Returns:
            np.array: Nx3 numpy array of generated poses where each pose contains the pose's
                [x, y, theta] values where theta is stored in radians.
        """
        return np.array((np.random.uniform(0.0, worldsize, num_poses),
                np.random.uniform(0.0, worldsize, num_poses),
                np.random.uniform(-np.pi, np.pi, num_poses))).T

    @staticmethod
    def originPoses(num_poses, worldsize):
        """Generates a numpy array of zero poses (i.e. all having the values [0, 0, 0])

        Args:
            num_poses (int): number of poses to generate
            worldsize  (float): specifies dimensions of the world (i.e. the 
                maximum values for the pose x and y values).
        
        Returns:
            np.array: Nx3 numpy array of generated poses where each pose contains the pose's
                [x, y, theta] values where theta is stored in radians.
        """
        return np.tile(
            np.array([worldsize/2.0, worldsize/2.0, np.pi/2]),
            (num_poses,1))

    @staticmethod
    def addOdometry(poses, odometry):
        """Applies odometry to input poses

        Applies the (non-probabilistic / deterministic) kinematic motion model from the
        CS427 particle filter notes by propagating the set of input poses according to 
        the input odometry. 

        Args:
            poses (np.array): Nx3 numpy array of poses where each pose contains the pose's
                [x, y, theta] values where theta is stored in radians.
            odometry (np.array): 1x3 or Nx3 numpy array of odometry parameters [dx, dy, dtheta].
                If the structure is 1x3 it applies the same odometry to all
                poses. If the structure is Nx3 the i'th vector in odometry is applied
                to the i'th vector in poses.

        Returns:
            np.array: Nx3 numpy array of updated poses where each pose contains the pose's
                [x, y, theta] values where theta is stored in radians. 
        """

        odometry = np.atleast_2d(odometry)

        # Extract the odometry parameters for delta x, delta y, and delta theta
        dx, dy, dtheta = odometry[:, 0], odometry[:, 1], odometry[:, 2]

        # Update x and y positions considering the current orientation
        poses[:, 0] += np.cos(poses[:, 2]) * dx - np.sin(poses[:, 2]) * dy
        poses[:, 1] += np.sin(poses[:, 2]) * dx + np.cos(poses[:, 2]) * dy

        # Update orientation (theta), and normalize angles to be within (-pi, pi]
        for i in range(len(poses)):
            poses[i, 2] += dtheta[i]
            poses[i, 2] = Pose2D.normaliseAngle(poses[i, 2])  # Apply normaliseAngle individually to each angle

        return poses


    @staticmethod
    def normaliseAngle(angle):
        """Normalises the input angle to the range ]-pi/2, pi/2]"""        
        if ((angle < np.pi) and (angle >= -np.pi)):
            return angle
        pi2 = np.pi*2
        nangle = angle - (int(angle/(pi2))*(pi2))
        if (nangle >= np.pi):
            return nangle - pi2
        elif (nangle < -np.pi):
            return nangle + pi2
        return nangle
