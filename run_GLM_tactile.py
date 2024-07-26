import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
from nilearn.plotting import plot_stat_map
from nilearn.image import mean_img

# File paths
func_files = [
    "/Users/yan/Documents/Project/afmri_rerun/tactile/pre_img_18.nii",
    "/Users/yan/Documents/Project/afmri_rerun/tactile/pre_img_19.nii",
    "/Users/yan/Documents/Project/afmri_rerun/tactile/pre_img_20.nii",

]

mask_file = "/Users/yan/Documents/Project/mask.nii"
# Define the experimental design
n_volumes = 6 + (10 + 10) * 7
frame_times = np.arange(n_volumes) * 3.0  # Assuming TR=3.0s

onsets_tactile = np.arange(6, n_volumes, 20)

duration_tactile = np.full(onsets_tactile.shape, 30)

# Create events dataframe
events_tactile = pd.DataFrame({
    'onset': onsets_tactile * 3.0,  # Convert volume index to time
    'duration': duration_tactile,
    'trial_type': 'tactile'
})



events = pd.concat([events_tactile])

contrasts = {
    'tactile': np.array([1, 0]),
}

# List to store individual contrast maps
individual_contrast_maps = {contrast_id: [] for contrast_id in contrasts.keys()}

# Fit model for each run individually and compute contrasts
for i, func_file in enumerate(func_files):
    print(f"Processing file {i+1}/{len(func_files)}: {func_file}")
    filenumber = func_file.split('_')[-1].split('.')[0]


    # Create the first-level model
    first_level_model = FirstLevelModel(
        t_r=3.0,
        slice_time_ref=0,
        mask_img=mask_file,
        noise_model='ar1',
        standardize=True,
        hrf_model='spm'
    )

    # Fit the model to the current functional file
    first_level_model = first_level_model.fit(func_file, events=events)

    for contrast_id, contrast_val in contrasts.items():
        z_map = first_level_model.compute_contrast(contrast_val, output_type='z_score')
        individual_contrast_maps[contrast_id].append(z_map)
        z_map.to_filename(f'{filenumber}_activation_map_run{i+1}_tactile.nii.gz')

        plot_stat_map(
            z_map, bg_img=mean_img(func_files), threshold=3.0,
            title=f'Run {i+1} Contrast: {filenumber}'
        )

# Compute group level maps
for contrast_id, z_maps in individual_contrast_maps.items():
    mean_z_map = mean_img(z_maps)
    mean_z_map.to_filename(f'{contrast_id}_activation_map_group_tactile.nii.gz')

    plot_stat_map(
        mean_z_map, bg_img=mean_img(func_files), threshold=3.0,
        title=f'Group Contrast: {contrast_id}'
    )

print("GLM analysis complete.")