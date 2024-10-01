import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json 
# import tf.transformations as tft
from scipy.spatial.transform import Rotation

def create_pyramid(position, rotation_matrix):
    # Define the base vertices of the pyramid (centered at origin)
    base_size = 1
    height = 1
    base_vertices = np.array([
        [-base_size*2, -base_size, height],
        [base_size*2, -base_size, height],
        [base_size*2, base_size, height],
        [-base_size*2, base_size, height]
    ])
    
    # Apex of the pyramid
    apex = np.array([0, 0, 0])
    
    # Apply rotation to the vertices
    rotated_base = base_vertices @ rotation_matrix.T
    rotated_apex = apex @ rotation_matrix.T
    
    # Translate vertices to the given position
    translated_base = rotated_base + position
    translated_apex = rotated_apex + position
    
    return translated_base, translated_apex

def plot_pyramid(ax, base_vertices, apex):
    
    # Plot base
    base_faces = [[base_vertices[j] for j in range(4)]]
    base = Poly3DCollection(base_faces, color='cyan', alpha=0.6)
    ax.add_collection3d(base)
    
    # Plot sides
    for i in range(4):
        side_faces = [[apex, base_vertices[i], base_vertices[(i + 1) % 4]]]
        side = Poly3DCollection(side_faces, color='blue', alpha=0.6)
        ax.add_collection3d(side)
    
    # Plot the apex
    ax.scatter(*apex, color='red')
    
    # Setting the axes properties
    # ax.set_xlim([position[0] - 2, position[0] + 2])
    # ax.set_ylim([position[1] - 2, position[1] + 2])
    # ax.set_zlim([position[2] - 2, position[2] + 2])
    

# Example position and rotation matrix
# position = np.array([4698.563537444519, -2677.284274008165, 151.65316196494646])
# rotation_matrix = np.array([
#     [0.27594440459764674,-0.036435975106700436,-0.9604827459612557,],
#     [0.9560525190006095, -0.09261096205466546, 0.278184813783748],
#     [-0.09908716609046553, -0.9950354915405446, 0.00927922899961453]
# ])
# fn = './gazebo3_transformed/transforms.json'
# with open(fn,'r') as f:
#     data = json.load(f)

# frames = data['frames']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# T = np.array([
#     [0,0,-1],
#     [1,0,0],
#     [0,1,0]
# ])
# for i in range(0,len(frames),10):
#     frame = frames[i]
#     transform_matrix = np.array(frame['transform_matrix'])
#     rotation_matrix = transform_matrix[:3,:3]
#     position = transform_matrix[:3, 3]
#     # rotation_matrix = np.dot(T, rotation_matrix)
#     # position = np.dot(T, position)
#     base_vertices, apex = create_pyramid(position, rotation_matrix)
#     plot_pyramid(ax, base_vertices, apex)

x,y,z = np.array([5,5,5])
direction = np.array([
    0-x,
    0-y,
    0-z
])

direction = direction / np.linalg.norm(direction)
yaw = np.arctan2(direction[1], direction[0])
pitch = -np.arcsin(direction[2])
R = Rotation.from_euler('yzx',[np.pi/2, 0, np.pi/2]).as_matrix()
rotation_matrix = Rotation.from_euler('xyz',[0,pitch,yaw]).as_matrix()
rotation_matrix = rotation_matrix@R
# qw,qx,qy,qz = tft.quaternion_from_euler(0, pitch, yaw)
# transform

# transform_matrix = np.array(frame['transform_matrix'])
# rotation_matrix = Rotation.from_quat([qx,qy,qz,qw]).as_matrix()
position = np.array([x,y,z])
base_vertices, apex = create_pyramid(position, rotation_matrix)
plot_pyramid(ax, base_vertices, apex)
ax.plot([x,0],[y,0],[z,0])


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xlim([-10, 10])
ax.set_ylim([-10,10])
ax.set_zlim([-10,10])

plt.show()
