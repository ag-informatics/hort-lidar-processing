#!/usr/bin/env python

'''
Performance Tests

Author: Steven Doyle
Contact: doyle110@purdue.edu
Date: 11/04/2023

This file contains the definitions for functions used to calculate performance of the processing pipeline.

'''

import numpy as np
import pandas as pd
import gc
import open3d as o3d
import os


def merge_pc(pcd1, pcd2):

    # Combine the points from both point clouds
    combined_points = o3d.utility.Vector3dVector(pcd1.points)
    combined_points.extend(pcd2.points)

    # Combine the colors from both point clouds
    combined_colors = o3d.utility.Vector3dVector(pcd1.colors)
    combined_colors.extend(pcd2.colors)

    # Create a new point cloud with the combined points and colors
    merged_cloud = o3d.geometry.PointCloud()
    merged_cloud.points = combined_points
    merged_cloud.colors = combined_colors

    return merged_cloud

def merge_nonplant_pc(itrial_dir, date_dir, plot_id, otrial_dir):

    # Load the point clouds
    in1='{a}\{b}\{c}\sor_nonplant_{c}.ply'.format(a=itrial_dir, b=date_dir, c=plot_id)
    in2='{a}\{b}\{c}\colored_non_plant_{c}.ply'.format(a=itrial_dir, b=date_dir, c=plot_id)
    pcd1 = o3d.io.read_point_cloud(in1)
    pcd2 = o3d.io.read_point_cloud(in2)

    # Combine the points from both point clouds
    merged_cloud = merge_pc(pcd1, pcd2)

    # Save the merged point cloud
    out='{a}\{b}\{c}\combined_nonplant_{c}.ply'.format(a=otrial_dir, b=date_dir, c=plot_id)
    o3d.io.write_point_cloud(out, merged_cloud)

    del pcd1, pcd2, merged_cloud
    gc.collect()

    return

def merge_nonplant_subsample_pc(itrial_dir, date_dir, plot_id, otrial_dir):
    
    # Load non-plant subsampled point clouds
    cats=['ground', 'fruit', 'outlier']
    pcds=[]
    for i in cats:
        inf='{a}\{b}\{c}\{d}_{c}.ply'.format(a=itrial_dir, b=date_dir, c=plot_id, d=i)
        if os.path.isfile(inf) == True:
            pcd=o3d.io.read_point_cloud(inf)
            pcds.append(pcd)
        else:
            pass
    
    # Merge non-plant subsampled point clouds
    merged_subsample= o3d.geometry.PointCloud()
    for i in pcds:
        merged_subsample = merge_pc(merged_subsample, i)
        
    # Save merged non-plant point cloud
    out='{a}\{b}\{c}\subsample_nonplant_{c}.ply'.format(a=otrial_dir, b=date_dir, c=plot_id)
    o3d.io.write_point_cloud(out, merged_subsample)

    del pcds, merged_subsample
    gc.collect()

    return

def iou(sample, gtrial_dir, ptrial_dir, date_dir, plot_id):

    # load point clouds
    pplant = o3d.io.read_point_cloud('{a}\{b}\{c}\sor_plant_{c}.ply'.format(a=ptrial_dir, b=date_dir, c=plot_id))
    pnonplant = o3d.io.read_point_cloud('{a}\{b}\{c}\combined_nonplant_{c}.ply'.format(a=ptrial_dir, b=date_dir, c=plot_id))
    gplant = o3d.io.read_point_cloud('{a}\{b}\{c}\plant_{c}.ply'.format(a=gtrial_dir, b=date_dir, c=plot_id))
    gnonplant = o3d.io.read_point_cloud('{a}\{b}\{c}\subsample_nonplant_{c}.ply'.format(a=gtrial_dir, b=date_dir, c=plot_id))

    # get number of points in the clouds
    gptotal = len(gplant.points)
    gntotal = len(gnonplant.points)

    # get number of points in test clouds
    pptotal = len(pplant.points)
    pntotal = len(pnonplant.points)

    # Create arrays
    pppoints = np.asarray(pplant.points)
    pnpoints = np.asarray(pnonplant.points)
    gppoints = np.asarray(gplant.points)
    gnpoints = np.asarray(gnonplant.points)

    # Create dfs
    cols=['x','y','z']
    ppdf=pd.DataFrame(pppoints, columns=cols)
    ppdf.drop_duplicates(keep='first', inplace=True)
    pndf=pd.DataFrame(pnpoints, columns=cols)
    pndf.drop_duplicates(keep='first', inplace=True)
    gpdf=pd.DataFrame(gppoints, columns=cols)
    gpdf.drop_duplicates(keep='first', inplace=True)
    gndf=pd.DataFrame(gnpoints, columns=cols)
    gndf.drop_duplicates(keep='first', inplace=True)

    # Calculate intersections
    ppgpintdf=pd.merge(ppdf, gpdf, how='inner', on=['x','y','z'])
    pngnintdf=pd.merge(pndf, gndf, how='inner', on=['x','y','z'])
    ppgnintdf=pd.merge(ppdf, gndf, how='inner', on=['x','y','z'])
    pngpintdf=pd.merge(pndf, gpdf, how='inner', on=['x','y','z'])

    ppgpint = len(ppgpintdf)
    pngnint = len(pngnintdf)
    ppgnint = len(ppgnintdf)
    pngpint = len(pngpintdf)

    # Calculate unions
    punion=ppgpint+ppgnint+pngpint
    nunion=pngnint+ppgnint+pngpint

    # Calculate IOUs
    piou = ppgpint/punion
    niou = pngnint/nunion

    # create a df with the descriptive stats
    processed_points = pptotal+pntotal
    groundtruthed_points = gptotal+gntotal
    cols=['sample','groundtruthed_total', 'processed_total','groundtruthed_plant_total',
        'groundtruthed_nonplant_total','processed_plant_total','processed_nonplant_total',
        'ppgpint','pngnint','ppgnint','pngpint','punion','nunion','piou','niou']
    df = pd.DataFrame(columns=cols)
    df.loc[0] = [sample,groundtruthed_points,processed_points,gptotal,gntotal,pptotal,
                pntotal,ppgpint,pngnint,ppgnint,pngpint,punion,nunion,piou,niou]
    
    del pplant, pnonplant, gplant, gnonplant, pppoints, pnpoints, gppoints, gnpoints, ppdf, pndf, gpdf, gndf, ppgpintdf, pngnintdf, ppgnintdf, pngpintdf
    gc.collect()
    
    return df