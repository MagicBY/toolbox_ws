import swift
import roboticstoolbox as rtb
import spatialmath as sm
from spatialmath import SE3
import numpy as np
import qpsolvers as qp




# Launch the simulator Swift
env = swift.Swift()
env.launch()


# Create a UR3 robot object
ur3 = rtb.models.UR3()


# Initial configuration
ur3.q = [0, -np.pi/2, -np.pi/6, -np.pi/6, -np.pi/6, 0]


# Add the UR3 to the simulator
env.add(ur3)
