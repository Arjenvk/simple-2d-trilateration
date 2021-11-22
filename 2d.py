import re
from easy_trilateration.least_squares import *
from easy_trilateration.graph import *
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_distance_2d(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

# Define UAS position, as a single position or array with sequenting oisitions
UAS_position = [[-5, 0],[-4, 0],[-3, 0],[-2, 0],[-1, 0]]

# define locations of Beacons
Beacon_1 = [-5, -5, 0]
Beacon_2 = [-5, 5, 0]
Beacon_3 = [5, -5, 0]
Beacon_4 = [5, 5, 0]

# create error list
errors = []
# start main loop
for position in UAS_position:

    # calculate distance debtween UAS and beacons, add to Beacon-object
    Beacon_1[2] = (calculate_distance_2d(position[0], position[1], Beacon_1[0], Beacon_1[1]))
    Beacon_2[2] = (calculate_distance_2d(position[0],position[1],Beacon_2[0],Beacon_2[1]))
    Beacon_3[2] = (calculate_distance_2d(position[0],position[1],Beacon_3[0],Beacon_3[1]))
    Beacon_4[2] = (calculate_distance_2d(position[0],position[1],Beacon_4[0],Beacon_4[1]))

    # create random noise, add noise to distance calculation
    mu, sigma = 0, 0.1 # mean and standard deviation
    noise = np.random.normal(mu, sigma, 4)

    Beacon_1[2] = Beacon_1[2] + noise[0]
    Beacon_2[2] = Beacon_2[2] + noise[1]
    Beacon_3[2] = Beacon_3[2] + noise[2]
    Beacon_4[2] = Beacon_4[2] + noise[3]

    # determine estimate position by leats squares
    arr = [Circle(Beacon_1[0], Beacon_1[1], Beacon_1[2] + noise[0]),
        Circle(Beacon_2[0], Beacon_2[1], Beacon_2[2] + noise[1]),
        Circle(Beacon_3[0], Beacon_3[1], Beacon_3[2] + noise[2]),
        Circle(Beacon_4[0], Beacon_4[1], Beacon_4[2] + noise[3]),
       ]
    result, meta = easy_least_squares(arr)
    print(result)

    # draw a plot
    create_circle(result, target=True)
    # draw(arr)

    # recover estimate position
    test = re.findall(r"[-+]?\d*\.\d+|\d+", str(result))
    x = float(test[0])
    y = float(test[1])

    # verschil in positie
    delta = calculate_distance_2d(position[0], position[1], x, y)
    print(delta)
    errors.append(delta)


print(errors)
plt.hist(errors)
plt.show()