#!/usr/bin/env python
import os
import math
import shutil
import numpy as np
import pandas as pd
from map_generation.MapGenerator import MapGenerator

def get_map(resolution=0.05, map_width=200, map_height=200, seed=16, ground_clearance_m=0.05, max_rock_size=10, cfa=0.02, crater_diameters=None, crater_densities=None, output_path=''):
    # cfa adjusts the baseline rock density

    # # Crater specifications (optional)
    # crater_diameters = [1.65, 1.35] # [0.65, 1.35]
    # crater_densities = [0.15, 0.05]

    # --- Output folder ---
    output_dir = output_path + "map"
    os.makedirs(output_dir, exist_ok=True)

    # raw image is saved as "RAW_map_{seed}.png"
    temp_png = f"map_{seed}.png"
    temp_raw_png = "RAW_" + temp_png

    # --- Map limits ---
    x_limits = [0, map_width * resolution]
    y_limits = [0, map_height * resolution]

    # --- Create MapGenerator instance ---
    mg = MapGenerator(cfa=cfa, x_limits=x_limits, y_limits=y_limits, verbose=False)

    # No locations to avoid for now
    locations_to_avoid = [[], []]
    locations_to_avoid_in_map_frame = mg._initialize_rover_target_locations(
        locations_to_avoid[0], locations_to_avoid[1], 2 * 1.0
    )

    # --- Generate terrain map ---
    terrain_map = mg.create_terrain_map(
        cfa=cfa,
        seed=seed,
        use_surveyor_power_law=True,
        ground_clearance_m=ground_clearance_m,
        crater_diameters=crater_diameters,
        crater_densities=crater_densities,
        locations_to_avoid=locations_to_avoid_in_map_frame,
        max_rock_size=max_rock_size
    )

    # --- Generate grid and save PNG ---
    # This call will save two images:
    #   1. The grid image as temp_png (e.g. "map_16.png")
    #   2. A raw version as "RAW_map_16.png"
    grid = terrain_map.generate_grid_mpl(resolution=resolution, map_file_name=temp_png)

    # --- Move files into the output folder ---
    # Move the main image:
    dest_png = os.path.join(output_dir, temp_png)
    shutil.move(temp_png, dest_png)

    # Move and rename the raw image to "raw_map_{seed}.png"
    dest_raw_png = os.path.join(output_dir, f"raw_{temp_png}")
    if os.path.exists(temp_raw_png):
        shutil.move(temp_raw_png, dest_raw_png)
    else:
        print("Warning: Raw image not found.")

    # --- Save grid as CSV ---
    csv_file = os.path.join(output_dir, f"map_mc_{seed}.csv")
    pd.DataFrame(np.asarray(grid)).to_csv(csv_file, sep=',', index=False, header=False)


    print("Map generated and saved:")
    print("PNG:", dest_png)
    print("Raw PNG:", dest_raw_png)
    print("CSV:", csv_file)
    
    return terrain_map, grid

if __name__ == "__main__":
    get_map()
