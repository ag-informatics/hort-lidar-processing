#!/usr/bin/env python

'''
Volume Computations

Author: Steven Doyle
Contact: doyle110@purdue.edu
Date: 10/15/2023

This file contains the definitions for functions used to compute volume metrics from processed plant point clouds

'''

# Import libraries
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull

def compute_voxel_vol(inf):
    # Read file
    pcd = o3d.io.read_point_cloud(inf)
    
    # Voxelization:
    vsizes=[0.02, 0.01, 0.005, 0.002, 0.001]
    volumes=[]
    for vsize in vsizes:
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=vsize)
        # Calculate the occupied volume:
        occupied_voxels = voxel_grid.get_voxels()  # Get the coordinates of occupied voxels
        voxel_volume = vsize**3
        volume = len(occupied_voxels) * voxel_volume
        volumes.append(volume)
    
    # Calculate the occupied volume:
    occupied_voxels = voxel_grid.get_voxels()  # Get the coordinates of occupied voxels
    voxel_volume = vsize**3
    volume = len(occupied_voxels) * voxel_volume

    # Get point cloud coordinates
    points = np.asarray(pcd.points)
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)

    # Calculate width, length, and height
    width = max_coords[0] - min_coords[0]
    depth = max_coords[1] - min_coords[1]
    height = max_coords[2] - min_coords[2]

    # Visualization
    #o3d.visualization.draw_geometries([voxel_grid])

    return width, depth, height, volumes



def compute_ch_vol(inf, outf):
    # Read file
    pcd = o3d.io.read_point_cloud(inf)

    # Compute convex hull for visualization
    hull, _ = pcd.compute_convex_hull()

    # Shapes
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))
    
    # Visualization
    #o3d.visualization.draw_geometries([pcd, hull_ls])

    # Convert the Open3D triangle mesh to numpy arrays
    mesh_vertices = np.asarray(hull.vertices)
    mesh_triangles = np.asarray(hull.triangles)

    # Create a scipy ConvexHull object
    convex_hull = ConvexHull(mesh_vertices)

    # Calculate the volume of the convex hull
    volume = convex_hull.volume

    # Compute the cross-sectional area of the convex hull
    projected_points = np.asarray(pcd.points)[:, :2]
    # Calculate convex hull area using scipy
    hull2 = ConvexHull(projected_points)
    cross_sectional_area = hull2.volume

    # Save the convex hull as a triangle mesh
    o3d.io.write_triangle_mesh("{a}.obj".format(a=outf), hull)

    return volume, cross_sectional_area