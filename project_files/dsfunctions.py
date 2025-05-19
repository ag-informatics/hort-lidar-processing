#!/usr/bin/env python

'''
Descriptive Statistics Functions

Author: Steven Doyle
Contact: doyle110@purdue.edu
Date: 12/17/2023

This file contains the definitions for functions used to generate descriptive statistics for plant point clouds.

'''

import pandas as pd
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull

def get_npts (pcd):
    # Convert to array and get number of points
    npts = len(np.asarray(pcd.points))
    return npts

def get_nnd (pcd):
    # Get average nearest neighbor distance
    points = np.asarray(pcd.points)

    def compute_nearest_neighbor_distance (points, pcd):
        distances = []
        kdtree = o3d.geometry.KDTreeFlann(pcd)

        for i in range(len(points)):
            [_, indices, _] = kdtree.search_knn_vector_3d(points[i], 2)  # Find 2 nearest neighbors (including self)
            distances.append(np.linalg.norm(points[i] - points[indices[1]]))  # Compute distance to the nearest neighbor
    
        return distances
    
    nnd = compute_nearest_neighbor_distance(points, pcd)
    # Get average nearest neighbor distance
    nn = np.mean(nnd)
    return nn

def get_ch_vol(pcd):
    # Get convex hull volume
    # Compute convex hull for visualization
    hull, _ = pcd.compute_convex_hull()

    # Convert the Open3D triangle mesh to numpy arrays
    mesh_vertices = np.asarray(hull.vertices)

    # Create a scipy ConvexHull object
    convex_hull = ConvexHull(mesh_vertices)

    # Calculate the volume of the convex hull
    volume = convex_hull.volume

    return volume