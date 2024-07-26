import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
from nilearn.plotting import plot_stat_map
from nilearn.image import mean_img

# File paths
func_files = [
    "/Users/yan/Documents/Project/afmri_rerun/blue/pre_img_16.nii",
    "/Users/yan/Documents/Project/afmri_rerun/blue/pre_img_07.nii",
    "/Users/yan/Documents/Project/afmri_rerun/blue/pre_img_10.nii",
    "/Users/yan/Documents/Project/afmri_rerun/blue/pre_img_11.nii",
    "/Users/yan/Documents/Project/afmri_rerun/blue/pre_img_13.nii",
    "/Users/yan/Documents/Project/afmri_rerun/blue/pre_img_14.nii"
]


mask_file = "/Users/yan/Documents/Project/mask.nii"

# Define the experimental design
n_volumes = 6 + 12 * (2 + 8) * 2
frame_times = np.arange(n_volumes) * 3.0  # Assuming TR=2.0s

onsets_blue = np.arange(6, n_volumes, 20)
onsets_green = np.arange(16, n_volumes, 20)

duration_blue = np.full(onsets_blue.shape, 6.0)
duration_green = np.full(onsets_green.shape, 6.0)

# Create events dataframe
events_blue = pd.DataFrame({
    'onset': onsets_blue * 3.0,  # Convert volume index to time
    'duration': duration_blue,
    'trial_type': 'blue'
})

events_green = pd.DataFrame({
    'onset': onsets_green * 3.0,  # Convert volume index to time
    'duration': duration_green,
    'trial_type': 'green'
})

events = pd.concat([events_blue, events_green])

contrasts = {
    'blue': np.array([1, 0]),
    'green': np.array([0, 1])
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
        z_map.to_filename(f'{contrast_id}_activation_map_run{filenumber}.nii.gz')

        plot_stat_map(
            z_map, bg_img=mean_img(func_files), threshold=3.0,
            title=f'Run {filenumber} Contrast: {contrast_id}'
        )

# Compute group level maps
for contrast_id, z_maps in individual_contrast_maps.items():
    mean_z_map = mean_img(z_maps)
    mean_z_map.to_filename(f'{contrast_id}_activation_map_group.nii.gz')

    plot_stat_map(
        mean_z_map, bg_img=mean_img(func_files), threshold=3.0,
        title=f'Group Contrast: {contrast_id}'
    )

print("GLM analysis complete.")