import numpy as np
from matplotlib.patches import Circle

class Obstacle:
    def __init__(self,radius,location):
        """[summary]
        Args:
            type ([type]): [description]
            radius ([type]): [description]
            location ([type]): [description]
        """
        self.radius = radius
        self.state = np.array([location[0],location[1],0])
