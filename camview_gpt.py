import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Quaternion Functions
def quaternion_product(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return (w, x, y, z)

def quaternion_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)

def transform_point_using_quaternion(point, quaternion):
    point_quaternion = (0,) + tuple(point)
    q_conj = quaternion_conjugate(quaternion)
    transformed = quaternion_product(quaternion_product(quaternion, point_quaternion), q_conj)
    return transformed[1:]

def transform_coordinates(translation, quaternion):
    vectors = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
    transformed_vectors = np.array([transform_point_using_quaternion(vector, quaternion) for vector in vectors]) + translation
    return transformed_vectors

# Transformation parameters
translation = [1, 2, 3]
quaternion = [1, 0, 0, 0]

transformed_coords = transform_coordinates(translation, quaternion)

# Visualization using Matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting the transformed camera coordinate vectors (R, G, B)
colors = ['r', 'g', 'b']
labels = ['X-axis', 'Y-axis', 'Z-axis']

for i, vector in enumerate(transformed_coords):
    ax.quiver(translation[0], translation[1], translation[2], 
              vector[0]-translation[0], vector[1]-translation[1], vector[2]-translation[2], 
              color=colors[i], label=labels[i], length=1, normalize=True)

ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.set_zlim([0, 5])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()