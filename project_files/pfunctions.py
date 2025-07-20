#!/usr/bin/env python

'''
Processing Functions

Author: Steven Doyle
Contact: doyle110@purdue.edu
Date: 10/15/2023

This file contains the definitions for functions used to process plant point clouds.

'''
import laspy
import numpy as np
from plyfile import PlyData, PlyElement
import gc
from skimage import color
import open3d as o3d


def las_to_ply(itrial_dir, date_dir, plot_id, otrial_dir):
    # Load file
    file = laspy.read('{a}\{b}\{c}\{c}.las'.format(a=itrial_dir,
                                                       b=date_dir,
                                                       c=plot_id))
    
    # Extract the XYZ coordinates and RGB colors
    xyz=np.vstack((file.x, file.y, file.z)).transpose()
    rgb = np.vstack((file.red, file.green, file.blue)).transpose()

    # Create a structured array for vertices
    vertices_dtype = [
        ('x', 'float32'),
        ('y', 'float32'),
        ('z', 'float32'),
        ('red', 'uint8'),
        ('green', 'uint8'),
        ('blue', 'uint8')
    ]

    # Normalize x axis around the center of the point cloud
    max_xyz = np.max(xyz, axis=0)
    min_xyz = np.min(xyz, axis=0)
    center = (max_xyz + min_xyz) / 2
    xyz[:, 0] -= center[0]

    # Assign point cloud array dimensions
    vertices = np.zeros(xyz.shape[0], dtype=vertices_dtype)
    vertices['x'] = xyz[:, 0]
    vertices['y'] = xyz[:, 1]
    vertices['z'] = xyz[:, 2]
    vertices['red'] = rgb[:, 0]
    vertices['green'] = rgb[:, 1]
    vertices['blue'] = rgb[:, 2]

    # Create the PlyElement with the vertex data
    vertex_element = PlyElement.describe(vertices, 'vertex')

    # Create the PlyData object and add the vertex element to it
    ply_data = PlyData([vertex_element])

    # Save the PlyData to a .ply file
    ply_data.write('{a}\{b}\{c}\{c}_v2.ply'.format(a=otrial_dir,
                                                b=date_dir,
                                                c=plot_id))
    del file, ply_data
    gc.collect()

    return



def convert_colors(itrial_dir, date_dir, plot_id, otrial_dir):
    # Load file
    point_cloud = o3d.io.read_point_cloud('{a}\{b}\{c}\{c}_v2.ply'.format(a=itrial_dir,
                                                                       b=date_dir,
                                                                       c=plot_id))
    
    # Extract RGB values from the point cloud
    rgb_colors = np.asarray(point_cloud.colors)

    # Convert RGB to HSV
    hsv_colors = color.rgb2hsv(rgb_colors)

    # Update the point cloud with the new colors
    point_cloud.colors = o3d.utility.Vector3dVector(hsv_colors)

    o3d.io.write_point_cloud("{a}\{b}\{c}\hsv_{c}_v2.ply".format(a=otrial_dir,
                                                              b=date_dir,
                                                              c=plot_id), point_cloud)

    del point_cloud
    gc.collect()

    return


def c_thresh_seg(itrial_dir, date_dir, plot_id, otrial_dir, h, s, v):
    # Load file
    pcd = o3d.io.read_point_cloud('{a}\{b}\{c}\hsv_{c}_v2.ply'.format(a=itrial_dir,
                                                                   b=date_dir,
                                                                   c=plot_id))

    # Convert string ranges to tuples
    h = h.split(',')
    h = tuple(map(float, h))
    s = s.split(',')
    s = tuple(map(float, s))
    v = v.split(',')
    v = tuple(map(float, v))

    # Define threshold values for HSV channels
    hue_threshold = h  # Range for hue channel
    saturation_threshold = s  # Range for saturation channel
    value_threshold = v  # Range for value channel

    # Apply thresholding to segment plants from non-plant objects
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Loop through points and determine if they are within plant thresholds or not
    plant_indices = []
    non_plant_indices = []
    for i, color in enumerate(colors):
        hsv = color[0], color[1], color[2]  # Extract HSV values from point cloud colors
        if hue_threshold[0] <= hsv[0] <= hue_threshold[1] and saturation_threshold[0] <= hsv[1] <= saturation_threshold[1] and value_threshold[0] <= hsv[2] <= value_threshold[1]:
            plant_indices.append(i)
        else:
            non_plant_indices.append(i)

    # Convert the lists to numpy arrays
    plant_indices = np.asarray(plant_indices)
    non_plant_indices = np.asarray(non_plant_indices)

    # Extract the plant points based on the segmented indices
    plant_points = points[plant_indices]
    plant_colors = colors[plant_indices]

    # If we wanted to do non-plant point clouds as well
    non_plant_points = points[non_plant_indices]
    non_plant_colors = colors[non_plant_indices]

    # Create new point cloud objects with only the plant and non-plant points
    plant_pcd = o3d.geometry.PointCloud()
    plant_pcd.points = o3d.utility.Vector3dVector(plant_points)
    plant_pcd.colors = o3d.utility.Vector3dVector(plant_colors)

    non_plant_pcd = o3d.geometry.PointCloud()
    non_plant_pcd.points = o3d.utility.Vector3dVector(non_plant_points)
    non_plant_pcd.colors = o3d.utility.Vector3dVector(non_plant_colors)

    # Extract HSV values from the point cloud
    hsv_colors = np.asarray(plant_pcd.colors)

    # Convert RGB to HSV
    from skimage import color # For some reason, this seems to need to be called every time
    hsv_colors = color.hsv2rgb(hsv_colors)

    # Update the point cloud with the new colors
    plant_pcd.colors = o3d.utility.Vector3dVector(hsv_colors)

    # Save point cloud
    o3d.io.write_point_cloud('{a}\{b}\{c}\colored_plant_{c}_v2.ply'.format(a=otrial_dir,
                                                                        b=date_dir,
                                                                        c=plot_id), plant_pcd)

    # Extract HSV values from the point cloud
    hsv_colors = np.asarray(non_plant_pcd.colors)

    # Convert RGB to HSV
    from skimage import color # For some reason, this seems to need to be called every time
    hsv_colors = color.hsv2rgb(hsv_colors)

    # Update the point cloud with the new colors
    non_plant_pcd.colors = o3d.utility.Vector3dVector(hsv_colors)

    # Save point cloud
    o3d.io.write_point_cloud('{a}\{b}\{c}\colored_non_plant_{c}_v2.ply'.format(a=otrial_dir,
                                                                            b=date_dir,
                                                                            c=plot_id), non_plant_pcd)

    del pcd, plant_pcd, non_plant_pcd
    gc.collect()

    return


def crop_b_box(itrial_dir, date_dir, plot_id, otrial_dir, width, depth, height):
    # Load file
    pcd = o3d.io.read_point_cloud('{a}\{b}\{c}\colored_plant_{c}_v2.ply'.format(a=itrial_dir,
                                                                            b=date_dir,
                                                                            c=plot_id))
    
    # Convert the point cloud to a NumPy array
    points = np.asarray(pcd.points)

    # Get the points below 0.00 because they will be outliers
    below_zero_z = points[:, 2] < 0.00

    # Filter the points below 0.00 on the z-axis
    filtered_points = points[below_zero_z]

    half_width = width / 2
    half_depth = depth / 2

    # Get the centroid of the point cloud
    centroid = np.mean(pcd.points, axis=0)

    # Calculate the bounding box coordinates based on the centroid
    min_x = centroid[0] - half_width
    max_x = centroid[0] + half_width
    min_y = centroid[1] - half_depth
    max_y = centroid[1] + half_depth
    min_z = 0  # Lower bound on the z-axis
    max_z = min_z + height

    # Crop the point cloud using the bounding box values
    cropped_cloud = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=(min_x, min_y, min_z),
                                                                 max_bound=(max_x, max_y, max_z)))

     # Save point cloud
    o3d.io.write_point_cloud('{a}\{b}\{c}\cropped_plant_{c}_v2.ply'.format(a=otrial_dir,
                                                                        b=date_dir,
                                                                        c=plot_id), cropped_cloud)

    del pcd, cropped_cloud
    gc.collect()

    return


def knn_outlier_removal(itrial_dir, date_dir, plot_id, otrial_dir, k, std_ratio):
    # Load file
    pcd = o3d.io.read_point_cloud('{a}\{b}\{c}\cropped_plant_{c}_v2.ply'.format(a=itrial_dir,
                                                                             b=date_dir,
                                                                             c=plot_id))

    # Convert point cloud to numpy arrays
    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors)

    # Create KDTree - this format is efficient for running knn tests
    tree = o3d.geometry.KDTreeFlann(pcd)

    # List to store inlier indices
    inlier_indices = []

    # Iterate over each point in the point cloud
    for i in range(len(xyz)):
        [_, idx, _] = tree.search_knn_vector_3d(xyz[i], k)
        neighbors_distances = np.linalg.norm(xyz[idx[1:], :] - xyz[i], axis=1)
        if np.std(neighbors_distances) < std_ratio:
            inlier_indices.append(i)

    # Filter point cloud using inlier
    filtered_xyz = xyz[inlier_indices, :]
    filtered_rgb = rgb[inlier_indices, :]

    # Create a new point cloud with the filtered data
    filtered_cloud = o3d.geometry.PointCloud()
    filtered_cloud.points = o3d.utility.Vector3dVector(filtered_xyz)
    filtered_cloud.colors = o3d.utility.Vector3dVector(filtered_rgb)

    # Calculate number of outliers removed
    #num_outliers_removed = len(xyz) - len(inlier_indices)
    #print("Number of outliers removed: ", num_outliers_removed)

    o3d.io.write_point_cloud('{a}\{b}\{c}\knn_plant_{c}_v2.ply'.format(a=otrial_dir,
                                                                    b=date_dir,
                                                                    c=plot_id), filtered_cloud)

    del pcd, filtered_cloud
    gc.collect()

    return

def sor_outlier_removal(itrial_dir, date_dir, plot_id, otrial_dir, k, std_ratio):
    # Load file
    pcd = o3d.io.read_point_cloud('{a}\{b}\{c}\cropped_plant_{c}_v2.ply'.format(a=itrial_dir,
                                                                             b=date_dir,
                                                                             c=plot_id))

    # Define outlier removal filter
    nb_neighbors = k
    std_ratio = std_ratio

    # Apply outlier removal filter
    pcd_filtered, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    o3d.io.write_point_cloud('{a}\{b}\{c}\sor_plant_{c}_v2.ply'.format(a=otrial_dir,
                                                                    b=date_dir,
                                                                    c=plot_id), pcd_filtered)

    # Create a new point cloud with only the outliers
    outlier_cloud = pcd.select_by_index(ind, invert=True)

    o3d.io.write_point_cloud('{a}\{b}\{c}\sor_nonplant_{c}_v2.ply'.format(a=otrial_dir,
                                                                       b=date_dir,
                                                                       c=plot_id), outlier_cloud)
    
    del pcd, pcd_filtered, outlier_cloud
    gc.collect()

    return

def crop_by_pc(rtrial_dir, date_dir, plot_id, ttrial_dir,):
    # Load raw and template point clouds
    raw = o3d.io.read_point_cloud('{a}\{b}\{c}\{c}_v2.ply'.format(a=rtrial_dir,
                                                               b=date_dir,
                                                               c=plot_id))
    template = o3d.io.read_point_cloud('{a}\{b}\{c}\cropped_plant_{c}_v2.ply'.format(a=ttrial_dir,
                                                                                  b=date_dir,
                                                                                  c=plot_id))
    
    # Find the minimum and maximum coordinates of the processed point cloud using the 
    # get_min_bound() and get_max_bound() functions.
    bbox_template = template.get_axis_aligned_bounding_box()

    # Use the `crop()` function to crop the raw point cloud using the calculated bounds.
    raw_cropped = raw.crop(bbox_template)

    o3d.io.write_point_cloud('{a}\{b}\{c}\cr_{c}_v2.ply'.format(a=rtrial_dir,
                                                             b=date_dir,
                                                             c=plot_id), raw_cropped)

    del raw, template, raw_cropped
    gc.collect()   

    return