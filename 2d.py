import re
from easy_trilateration.least_squares import *
from easy_trilateration.graph import *
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def calculate_distance_2d(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# Define UAS position, as a single position or array with sequenting positions
begin_positie = [-20, 0]
stappen = range(-20, 0)
print(stappen)

posities = []
for i in stappen:
    posities.append([i,0])

print(posities)

# define locations of Beacons, 3rd column for distances
Beacon_1 = [-5, -5, 0]
Beacon_2 = [-5, 5, 0]
Beacon_3 = [5, -5, 0]
Beacon_4 = [5, 5, 0]

# create estimate position and error list
estimate_positions = []
errors = []
# start main loop
for position in posities:

    # calculate distance debtween UAS and beacons, add to Beacon-object
    Beacon_1[2] = (calculate_distance_2d(position[0], position[1], Beacon_1[0], Beacon_1[1]))
    Beacon_2[2] = (calculate_distance_2d(position[0],position[1],Beacon_2[0],Beacon_2[1]))
    Beacon_3[2] = (calculate_distance_2d(position[0],position[1],Beacon_3[0],Beacon_3[1]))
    Beacon_4[2] = (calculate_distance_2d(position[0],position[1],Beacon_4[0],Beacon_4[1]))

    # create random noise, add noise to distance calculation to re-create range measurement
    mu, sigma = 0, 0.05 # mean and standard deviation
    noise = np.random.normal(mu, sigma, 4)

    Beacon_1[2] = Beacon_1[2] + noise[0]
    Beacon_2[2] = Beacon_2[2] + noise[1]
    Beacon_3[2] = Beacon_3[2] + noise[2]
    Beacon_4[2] = Beacon_4[2] + noise[3]

    # determine estimate position by least squares
    data = [Circle(Beacon_1[0], Beacon_1[1], Beacon_1[2]),
        Circle(Beacon_2[0], Beacon_2[1], Beacon_2[2]),
        Circle(Beacon_3[0], Beacon_3[1], Beacon_3[2]),
        Circle(Beacon_4[0], Beacon_4[1], Beacon_4[2]),
       ]
    result, meta = easy_least_squares(data)

    # draw a plot
    create_circle(result, target=True)
    draw(data)

    # recover estimate position
    test = re.findall(r"[-+]?\d*\.\d+|\d+", str(result))
    x = float(test[0])
    y = float(test[1])

    # voeg estimate positiions toe
    estimate_positions.append([x,y])
    # (absoluut) verschil in positie
    delta = calculate_distance_2d(position[0], position[1], x, y)

    # voeg verschil toe aan array met alle errors
    errors.append(delta)

# show positions and errors every step
print(estimate_positions)
print(errors)

# plot positions
plt.plot(posities)
plt.plot(estimate_positions, 'r')
orange_patch = mpatches.Patch(color='orange', label='True positions')
red_patch = mpatches.Patch(color='red', label='Estimated positions')
plt.legend(handles=[orange_patch, red_patch])
plt.xlim(-1, len(posities))
plt.ylim(-0.2, 0.2)
plt.xlabel("Sample")
plt.xticks(range(0,21))
plt.ylabel("")
plt.show()

# plot errors
plt.plot(errors, 'ro')
plt.xlim(-1, len(posities) )
plt.ylim( 0, 0.1)
plt.xlabel("Sample")
plt.xticks(range(0,21))
plt.ylabel("Absolute error in position [m]")
plt.show()


# statistical analysis or errors
mean = np.mean(errors)
deviation = np.std(errors)
print(mean)
print(deviation)


# plottting errors
test = np.subtract(posities, estimate_positions)
trans = np.transpose(test)
print(trans)

x = trans[0]
y = trans[1]



fig, ax_nstd = plt.subplots(figsize=(6, 6))
mu = 0, 0
scale = 8, 5


ax_nstd.scatter(x, y, s=0.5)

confidence_ellipse(x, y, ax_nstd, n_std=1,
                   label=r'$1\sigma$', edgecolor='firebrick')
confidence_ellipse(x, y, ax_nstd, n_std=2,
                   label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
confidence_ellipse(x, y, ax_nstd, n_std=3,
                   label=r'$3\sigma$', edgecolor='blue', linestyle=':')

ax_nstd.scatter(mu[0], mu[1], c='red', s=3)
ax_nstd.set_title('Different standard deviations')
ax_nstd.legend()
plt.xlabel("Positional error x-axis [m]")
plt.ylabel("Positional error Y-axis [m]")
plt.show()