import numpy as np
import cv2

from ParticleFilter2D.ParticleFilter2D import ParticleFilter2D
from ParticleFilter2D.SensorModel import SensorModel

class WorldDrawer:
    """Provides visualisation of WorldModel and ParticleFilter data

    Attributes:
        worldsize (float): size of x & y-dimensions of world in pixels.
        image (2D array): image for opencv to render and display world state.
    """
    def __init__(self, worldsize = 750):
        self.worldsize = worldsize
        self.image = np.zeros((int(worldsize), int(worldsize), 3))

        cv2.namedWindow("World", cv2.WINDOW_AUTOSIZE)

    def showImage(self):
        """Display image. Note you must call cv2.waitKey() for this to take affect."""
        cv2.imshow("World", self.image)

    def drawWorld(self, world):
        """Render current world state. Note: should be followed by showImage."""
        self.image = np.zeros((int(self.worldsize), int(self.worldsize), 3))
        self.drawParticle(world.currentPose.squeeze(), (0,0,255), 1)
        self.drawLandmarks(world)

    def drawLandmarks(self, world):
        """Draw the world map."""
        for c in range(world.landmarks.shape[0]):
            cv2.circle(self.image, 
                    (int(world.landmarks[c,0]), 
                    int(world.landmarks[c,1])),
                    5,(255,255,255))

        # highlight visible landmarks in red
        sm = SensorModel()
        for v in sm.getVisibleLandmarks(world.currentPose, world.landmarks):
            cv2.circle(self.image, 
                    (int(world.landmarks[v.landmark_index, 0]), 
                    int(world.landmarks[v.landmark_index, 1])),
                    6, (0, 0, 255))

    def drawParticles(self, particle_filter: ParticleFilter2D):
        """Draw particles as robot poses."""
        for p in particle_filter.poses:
            self.drawParticle(p)

    def drawParticle(self, p, color=(255,0,0), weight=1):
        """Draws an individual particle."""
        sm = SensorModel()
        cv2.circle(self.image, 
            (int(p[0]), int(p[1])),
            4, color, weight)
        cv2.line(self.image,
            (int(p[0]), int(p[1])),
            (int(p[0] + np.cos(p[2]) * 10), 
                int(p[1] + np.sin(p[2]) * 10)),
            color, weight)
        cv2.line(self.image,
            (int(p[0]), int(p[1])),
            (int(p[0] + np.cos(p[2]+sm.fov) * 10), 
                int(p[1] + np.sin(p[2]+sm.fov) * 10)),
            (255,0,255), weight)
        cv2.line(self.image,
            (int(p[0]), int(p[1])),
            (int(p[0] + np.cos(p[2]-sm.fov) * 10), 
                int(p[1] + np.sin(p[2]-sm.fov) * 10)),
            (255,0,255), weight)



    
