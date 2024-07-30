"""ParticleFilter2D driver script

This script provides driver code for setting up a simulated environment (i.e. WorldModel)
and ParticleFilter2D (and associated classes). It then display the environment and allows
the user to interact with it via the keys:
Movement:
'w': forward
's': backward
'a': rotate-left
'd': rotate-right
Other:
'q': quit
"""
from ParticleFilter2D import WorldModel, SensorModel, WorldDrawer, ParticleFilter2D

import numpy as np
import cv2

def keyboardHandler(key):
    """Returns np.array for odometry parameters based on the input key.
    
    Args:
        key (int): input character from {'w','a','s','d'} corresponding to forward, rotate-left, etc.

    Returns:
        np.array: [x, y, theta] corresponding to odometry given the input key (or [0, 0, 0] otherwise).
    """
    keycode = chr(key & 255)
    if (keycode == 'w'):
        return np.array([10, 0, 0], dtype='float64')
    elif (keycode == 's'):
        return -np.array([10, 0, 0], dtype='float64')
    elif (keycode == 'd'):
        return np.array([0, 0, np.pi*0.05], dtype='float64')
    elif (keycode == 'a'):
        return -np.array([0, 0, np.pi*0.05], dtype='float64')
    return np.array([0, 0, 0], dtype='float64')

def main():
    ## Simulation parameters
    NUM_LANDMARKS = 30
    NUM_PARTICLES = 150
    WORLDSIZE = 750

    ## Create simulated environment
    wm = WorldModel(WORLDSIZE)

    ## Add landmarks / map 
    landmarks = np.random.uniform(0,WORLDSIZE,(NUM_LANDMARKS,2))
    wm.addLandmarks(landmarks)

    ## Sensor model for generating measurements
    sm = SensorModel()

    ## Instantiate a particle filter
    pf = ParticleFilter2D(NUM_PARTICLES)
    
    ## Create visualisation
    key = 0
    wd = WorldDrawer(WORLDSIZE)

    ## Main loop
    while ('q' != chr(key & 255)):
        ## Propagate the robot
        odometry = keyboardHandler(key)
        wm.addOdometry(odometry)

        ## Generate measurements
        measurements = wm.sampleMeasurements(sm)
        
        ## Execute filter 
        pf.processFrame(odometry, measurements, wm.landmarks)

        ## Render world
        wd.drawWorld(wm)
        wd.drawParticles(pf)
        wd.showImage()

        key = cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
