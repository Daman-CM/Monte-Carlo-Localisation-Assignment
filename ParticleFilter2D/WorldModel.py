import numpy as np

from ParticleFilter2D.Pose2D import Pose2D

class WorldModel:
    """Provides model of simulated environment for use with ParticleFilter2D.

    Attributes:
        landmarks (array): Nx2 array of landmark [x, y] location vectors.
        currentPose (array): 1x3 array of pose [x, y, theta] values where theta is 
            stored in radians. 
    """
    def __init__(self, worldsize = 750.0):
        """Creates a simulated environment with no landmarks and a random starting pose.

        Args:
            worldsize (float): size of the world in pixels.
        """
        self.landmarks = np.array([])
        self.currentPose = Pose2D.randomPoses(num_poses=1, worldsize=worldsize)

    def addOdometry(self, odometry: Pose2D):
        """Applies odometry to currentPose

        Applies the (non-probabilistic / deterministic) kinematic motion model
        from the CS427 particle filter notes by propagating the current pose
        forward by the input odometry. 

        Args:
            odometry (np.array): 1x3 numpy array of odometry parameters [dx, dy, dtheta] 
        """ 
        self.currentPose = Pose2D.addOdometry(self.currentPose, odometry)

    def sampleMeasurements(self, sensor_model):
        """Returns an array of *noise corrupted* measurements to visible landmarks.

        Uses the sensor_model implemented in the SensorModel class to generate
        a set of measurements.

        Args:
            sensor_model (SensorModel): sensor model class

        Returns:
            list: list of *noise corrupted* SensorMeasurements to visible 
                landmarks.
        
        See also:
            SensorMeasurement.sampleMeasurements()
        """          
        return sensor_model.sampleMeasurements(self.currentPose, self.landmarks)

    def addLandmarks(self, landmarks):
        """Adds / Concatenates the landmarks to the current map.
        
        Args:
            landmarks (array): Nx2 array of landmark [x, y] location vectors.
        """
        if (landmarks.shape[1] != 2):
            print("landmarks must be N x 2")
            return
        
        if (self.landmarks.size == 0):
            self.landmarks = landmarks
            return

        self.landmarks = np.concatenate((self.landmarks, landmarks),
                                        axis=1)

    def clearLandmarks(self):
        """Clears the current map."""
        self.landmarks = np.array([])