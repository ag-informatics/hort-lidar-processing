

# Horticultural Crop LiDAR Processing
This repository serves to store the code used for processing point clouds of horticultural crops. It was last updated on May 19, 2025.
## File Locations
All plant point clouds and accompanying 2D images are stored in the Sample_LiDAR_Data folder on Purdue Box. The file setup of this repository mirrors that the user can simply copy this repository's files into a separate folder and add the point clouds, or add the point clouds to this repository in their correct location. Regardless, the point cloud files are much too large for GitHub to store.
## Repository Structure
project_folders
- Biomass_and_Volume_Regressions - stores regressions for parameters and volume calculations.
- Cucumber_Fertilizer_Trial - stores associated point clouds.
- Garden_Plots - stores associated point clouds.
- Pepper_Variety_Trial - stores associated point clouds.
- Personal-Sized_Watermelon_Variety_Trial - stores associated point clouds.
- plot_figs - stores figures produced from analyses.
- Seedless_Watermelon_Variety_Trial - stores associated point clouds.
- Tomato_Fertilizer_Trial - stores associated point clouds.
- Transplant-Age_Plants - stores associated point clouds.
- Watermelon_Irrigation_Trial - stores associated point clouds.
## Important Scripts
1. performance_test_pipeline.ipynb - conducts IoU tests for processing validation.
2. processing_pipeline.ipynb - processes raw point clouds into finalized plant and non-plant point clouds.
3. sensor_validation_pipeline.ipynb - calculates difference between sensed and measured plant height.
4. volume_biomass_regressions.ipynb - fits regression equations for parameters estimated from the calibration trial samples with volume metrics.
5. volume_pipeline.ipynb - calculates volume metrics for processed plant point clouds.