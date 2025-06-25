import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from matplotlib.animation import FFMpegWriter
import heapq
import time
import math
import copy

# PSE path planner: Docker
import docker
import tarfile
import io
import re

# PSE path planner: Python
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import csv

# Learning framework
from collections import deque
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Local
from get_map import get_map

# Log file

import sys
import atexit

class Logger(object):
    def __init__(self, filename='logfile.log'):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')
        atexit.register(self.close)  # Register the close method to be called when the program exits

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        if not self.log.closed:
            self.log.flush()

    def close(self):
        if not self.log.closed:
            self.log.close()

# Warnings
import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = warnings, 2 = errors, 3 = fatal


# --- ToDo ---
# Implement hot potato algorithm?

# Last point and first point are connected on trajectories plot. Check how this was avoided in CADRE PSE stack. This happenned before and was because trajectories were looped.
# We need to handle unreachable points (see gepeto)
# Do not mark a radio of cells as explored if they are an obstacle


# --- Hot Parameters ---
# Parameters that are changed more usually
video_options = ['display', 'save', 'no video']
video_option = video_options[1]
buffer_size = 300               # rover buffer size
num_rovers = 3                  # Number of rovers
max_neighbors = 2               # max number of neighbors to consider. The action space is max_neighbors+1, since 0 is hold
HTL = False                     # If True, the hops to lander are included in the state space
sim_duration = 5000             # in time steps
traj_type = 2                   # [0,1,2] [0: Random objectives and A* policy for trajectories, 1 objectives and trajectories generated with JPL's PSE code, 2: same as 1 but in a python script]
map_type = 1                    # [0,1] 0: Random rocks, 1 realistic maps generated with lunar distribution
comm_policy = 1                 # ['greedy', 'spray_wait', 'MA-DRL', 'GAT-RL']
import_trajectories = False     # If True, the trjectories and goals are not generated, but read from a csv
import_models = False           # If True, the models are not trained, but read from a folder. No exploration is done.
explore_train = True            # If True, the models are trained with exploration. If False, the models are just exploiting.

# --- General parameters ---
# num_rovers = 10                  # Number of rovers
num_nodes = num_rovers + 1      # Total nodes (including the lander)
threshold_rover = 0.4           # Communication range for rover-to-rover links
threshold_base = 0.4            # Higher communication range when one node is the lander
step_size = 0.005               # Step size along the planned path. Pnly when traj_type==0
noise_strength = 0.3            # Noise added to rover movement
movement_seed = 1               # movement random noise seed
wait = 0                        # rovers will wait this time steps after reaching an objective
noise_movement = False          # if True adds some noise to rovers movement
packet_coordinates = True       # if True, the explored area is the one associated with the packets that reached the lander, otherwise the explored area is the one where rovers were not congested
trajectories_local_path =       './Trajectories_goals/'
grd_size = 200                  # Map size. PSE objective generator will not generate trajectories under 142x142
paint_radius = 1                # Radius (in grid cells) around the rover to paint
resolution = 0.05               # Resolution for the ground grid map

# --- Exploration parameters ---
frontier_radius = 10

# --- Plot options ---
dropped_counter = True
show_ttl = True

# --- Map parameters ---
map_types = ['simple_map', 'real_map']
# map_type = 1                    # [0,1] 0: Random rocks, 1 realistic maps generated with lunar distribution
map_type_str = map_types[map_type]  # string used for the results output folder
if map_type==0:
    num_rock_clusters = 5           # Number of rock clusters
    rock_cluster_radius = 5         # Rock cluster size
    rock_seed = 6                   # Seed for random allocation of rocks
elif map_type==1:
    crater_diameters = [0.65, 1.35] # [0.65, 1.35]
    crater_densities = [0.15, 0.05] # [0.15, 0.05]
    cfa = 0.02                      
    ground_clearance_m = 0.08       # 0.05. Inverse proportional to the density of rocks
    max_rock_size = 50              # 10. Max rock size
    map_seed = 16                   # Seed for random realistic map generation
objective_seed = 1                  # Seed for random allocation of rover objectives

# --- Communication Parameters ---
generate_traffic = True         # If True, the packets are generated by rovers
packet_size = 128000            # Size of each packet in bits
# buffer_size = 150               # rover buffer size
packet_generation_flow = 1      # Generate a new packet every N steps
packets_per_objective = 0       # number of packets created when an objective is reached
tx_rate_direct = 5              # Direct transmission rate (packets per time interval)
tx_rate_indirect = 3            # Indirect transmission rate (packets per time interval)
comm_policies = ['Greedy', 'Spray_and_Wait', 'MA-DRL', 'GAT-RL']
# comm_policy = 1                   # ['greedy', 'spray_wait']
comm_policy_str = comm_policies[comm_policy]  # string used for the results output folder

# --- Docker parameters ---
image_name = "path-planner-image0.3"
traj_types = ['Random', 'PSE', 'PSE_python']
# traj_type = 1                                         # [0,1] 0: Dummy objectives and A* policy for trajectories, 1 objectives and trajectories generated with JPL's PSE code
traj_type_str = traj_types[traj_type]                   # string used for the results output folder
map_origin = -(grd_size * resolution)/2 #-2.5           # bottom left corner values in meters
fov = '6.28'                                            # Field of View (rads). 6.28 (2pi) means full vision.
map_width_m = grd_size * resolution                     # width in meters
lander_m = (3.5, 3.5)                                   # lander position in meters
lander_pose = f"{lander_m[0]} {lander_m[1]} 0.0"        # lander position

start_args = []                 # rovers starting position where lander
for _ in range(num_rovers):
    start_args.extend(["-s", lander_pose])
trajectories_path = "/CADRE/cadre-pse/simple-sim/build/exploration_trajectories.vnl"
trajectory_file_name = "exploration_trajectories.vnl"
local_trajectories_folder = '/Trajectories_goals/'


# --- Learning - General ---
# import_models = True          # If True, the models are not trained, but read from a folder. No exploration is done.
update_f = 1000                 # frequency of target network update
action_seed = 42                # Seed for random action selection
models_local_path = './Models/' # Path to save the models

# --- Learning - State space ---
# max_neighbors = 3   # max number of neighbors to consider. The action space is max_neighbors+1, since 0 is hold
state_size_node = 3 # State size per node
connected = 10      # Connected to lander. In Lander class this is 0
unconnected = 50    # Not connected to lander
pos_norm = 50       # Normalize position [0, pos_norm]
distance_norm = 50  # Normalized max distance to lander
ttl_norm = 1        # TTL value divided by this number to normalize it
no_ttl = 100         # No TTL value # FIXME this is is an arbitrary value. compute the max possible ttl for the current map?
hops_to_lander = 50 # Max hops to lander represented
max_buffer = 50     # Max buffer size normalized
unavailable = 100   # Value for unavailable neighbors
# HTL = True          # If True, the hops to lander are included in the state space
if HTL:
    state_size_node += 1 # Add hops to lander to the state size

# --- Learning - Rewards ---
penalty_unavailable = -100
penalty_hold        = 0
penalty_forward     = 0
penalty_drop        = -50
reward_deliver      = 100
buffer_steppness    = 3     # Steepness of the exponential buffer penalty. If bigger the difference between a full and an empty buffer is bigger

# --- Option for video generation ---
# video_options = ['display', 'save', 'no video']
# video_option = video_options[1]

# --- Time Granularity and Duration ---
time_interval = 1                  # Simulation time (in seconds) per frame
# sim_duration = 4000                 # in time steps
frames_count = int(sim_duration / time_interval)

# --- Colormap for Rover Load ---
colors = ["#ADD8E6", "#FF0000"]  # Light blue to red.
load_cmap = mcolors.LinearSegmentedColormap.from_list("load_cmap", colors)

# --- Create results folder if it does not exist ---
outputPath = f"./Results/{comm_policy_str}_{map_type_str}_{grd_size}_{traj_type_str}_{sim_duration}s_{num_rovers}_r/"
os.makedirs(outputPath, exist_ok=True)

# =============================================================================
# General functions
# =============================================================================
def meters_to_normalized(x_m, y_m, map_origin, grd_size, resolution):
    map_width_m = grd_size * resolution
    x_n = (x_m - map_origin) / map_width_m
    y_n = 1.0 - ((y_m - map_origin) / map_width_m)  # invert Y
    return x_n, y_n

def convert_meters_to_normalized(traj_data, map_origin, grd_size, resolution):
    total_size_meters = grd_size * resolution
    normalized_traj = traj_data.copy()

    # Convert x, y positions from meters to normalized coordinates
    for i in range(traj_data.shape[1]//3):
        x_idx = 2 + 3*i
        y_idx = 3 + 3*i

        normalized_traj[:, x_idx] = (traj_data[:, x_idx] - map_origin) / total_size_meters
        normalized_traj[:, y_idx] = (traj_data[:, y_idx] - map_origin) / total_size_meters

    return normalized_traj

def assign_docker_trajectories_and_goals(surface, outputPath):
    """
    Calls the PSE planner, reads trajectories and timed objectives,
    converts them to normalized [0,1] coordinates, and stores them in the surface's rovers.
    """
    traj_data_meters, objectives_dict_timed = get_docker_trajectories(outputPath)
    grd_size = surface.grd_size
    height = grd_size * resolution

    # --- Remove NaNs ---
    num_agents = int(traj_data_meters[0, 1])
    cleaned_trajs = []

    for i in range(num_agents):
        x_idx = 2 + 3 * i
        y_idx = 3 + 3 * i

        # Extract x, y, and mask valid positions (i.e., not NaN)
        x = traj_data_meters[:, x_idx]
        y = traj_data_meters[:, y_idx]
        valid_mask = ~np.isnan(x) & ~np.isnan(y)

        # Keep only valid (non-NaN) rows
        cleaned_traj = traj_data_meters[valid_mask, :]
        cleaned_trajs.append(cleaned_traj)

    # --- Normalize to [0,1] ---
    # traj_data_normalized = traj_data_filled.copy()
    traj_data_normalized = traj_data_meters.copy()
    num_agents = int(traj_data_normalized[0, 1])
    for i in range(num_agents):
        x_idx = 2 + 3 * i
        y_idx = 3 + 3 * i
        traj_data_normalized[:, x_idx] = (traj_data_meters[:, x_idx] - map_origin) / (grd_size * resolution)
        traj_data_normalized[:, y_idx] = (traj_data_meters[:, y_idx] - map_origin) / (grd_size * resolution)

    # np.savetxt(
    #     os.path.join(outputPath, "debug_normalized_trajectories.csv"),
    #     traj_data_normalized,
    #     delimiter=",",
    #     header=",".join([f"time_{i},x_{i},y_{i}" for i in range(num_agents)]),
    #     comments=""
    # )

    for i, rover in enumerate(surface.rovers):
        rover.trajectory = list(zip(
            traj_data_normalized[:, 2 + 3 * i],
            traj_data_normalized[:, 3 + 3 * i]
        ))

        # Timed goals (normalize from meters to [0,1])
        timed_goals_meters = objectives_dict_timed.get(i, [])
        rover.timed_goals = []
        for (timestep, gx, gy) in timed_goals_meters:
            gx_norm = (gx - map_origin) / (grd_size * resolution)
            gy_norm = (gy - map_origin) / (grd_size * resolution)
            rover.timed_goals.append((timestep, gx_norm, gy_norm))

        rover.timed_goals_index = 0
        
    surface.export_rover_trajectories(f'{outputPath}/Trajectories_goals/')
    surface.export_rover_goals(f'{outputPath}/Trajectories_goals/')

def get_docker_trajectories(outputPath):
    '''
    Runs Docker container for the PSE path planner, returning:
      (traj_data, objectives_dict_timed)
    Where:
      - traj_data is from exploration_trajectories.vnl (with Y flipped).
      - objectives_dict_timed is a dict:
            agent_id -> list of (docker_time, x_m, y_m)
        Each entry is the time from Docker plus the (x, y) in meters.
    '''
    print(f'Initializing Docker container with image: {image_name}')
    client = docker.from_env()

    container = client.containers.run(
        image=image_name,
        volumes={
            os.path.abspath(outputPath + 'map/'): {
                "bind": "/mnt/maps",
                "mode": "ro"
            }
        },
        working_dir="/CADRE/cadre-pse/simple-sim/build",
        detach=True,
        tty=True,
        stdin_open=True,
        remove=False
    )

    exit_code, _ = container.exec_run("mkdir -p ./data_products")
    if exit_code != 0:
        container.remove(force=True)
        raise RuntimeError("Failed to create ./data_products inside container")

    # ---------------------------------------------------------------
    # Run the planner, capturing stdout
    # ---------------------------------------------------------------
    print(f'Generating trajectories for {num_rovers} rovers...')
    exit_code, output = container.exec_run(
        cmd=[
            "./bin/simple-sim-explorer-cli",
            "-m", "/mnt/maps/map_mc_16.csv",
            "-w", str(grd_size),
            "-h", str(grd_size),
            "-r", str(resolution),
            "-x", f'{map_origin} {map_origin}',
            *start_args,
            "-u", "",
            "-t", str(sim_duration),
            '-a', fov
        ],
        stdout=True,
        stderr=True
    )

    decoded_out = output.decode()
    # print("Planner output lines:\n", decoded_out)

    # We'll parse "Time 418" lines, then the following line "New goal for agent..."
    # and associate them. The pattern is:
    # Time\s+(\d+)
    # New goal for agent\s+(\d+)\s+is at\s+([-\d\.]+)\s+([-\d\.]+)\s+[.\d-]+
    # state machine approach:
    lines = decoded_out.splitlines()
    last_time = None
    objectives_dict_timed = {}  # agent_id -> list of (docker_time, x_m, y_m)

    height = grd_size * resolution

    for line in lines:
        # 1) Check if line is 'Time ###'
        match_time = re.match(r"^Time\s+(\d+)", line)
        if match_time:
            last_time = int(match_time.group(1))
            continue

        # 2) Check if line says 'New goal for agent X is at x y z'
        #    associate the previously captured last_time
        match_goal = re.search(
            r"New goal for agent\s+(\d+)\s+is at\s+([-\d\.]+)\s+([-\d\.]+)\s+[-\d\.]+",
            line
        )
        if match_goal and last_time is not None:
            agent_id = int(match_goal.group(1))
            x_m = float(match_goal.group(2))
            y_m = float(match_goal.group(3))

            # Flip Y if desired
            y_m = 2 * map_origin + height - y_m

            objectives_dict_timed.setdefault(agent_id, []).append((last_time, x_m, y_m))

    print('Trajectories generated!')

    # ---------------------------------------------------------------
    # 2) Read the trajectory file
    # ---------------------------------------------------------------
    stream, _ = container.get_archive(trajectories_path)
    file_obj = io.BytesIO()
    for chunk in stream:
        file_obj.write(chunk)
    file_obj.seek(0)

    with tarfile.open(fileobj=file_obj) as tar:
        trajectory_file = tar.extractfile(trajectory_file_name)
        traj_data = np.loadtxt(io.StringIO(trajectory_file.read().decode()), skiprows=1)

    container.stop()
    container.remove()
    print('Docker container deleted')

    # ---------------------------------------------------------------
    # 3) Flip Y in trajectory data as well
    # ---------------------------------------------------------------
    num_agents = int(traj_data[0, 1])
    for i in range(num_agents):
        y_idx = 3 + 3*i
        traj_data[:, y_idx] = 2 * map_origin + height - traj_data[:, y_idx]

    print(f'Trajectories read from Docker: {trajectories_path}/{trajectory_file_name}')
    return traj_data, objectives_dict_timed

def plot_final_trajectories(surface, output_dir='.'):
    print('Plotting rovers trajectories...')
    # Map data
    obstacle_map = (surface.ground >= 100).astype(np.uint8)

    # Origin and extent
    origin = (map_origin, map_origin)  # consistent with map_origin
    grd_size = surface.grd_size
    height = grd_size * resolution
    extent = [
        origin[0],
        origin[0] + grd_size * resolution,
        origin[1],
        origin[1] + height
    ]

    # Plot map and trajectories
    plt.figure(figsize=(10, 10))
    plt.imshow(obstacle_map, cmap="gray_r", origin="lower", extent=extent)

    for rover in surface.rovers:
        trajectory = np.array(rover.trajectory)
        trajectory_m = trajectory * grd_size * resolution + map_origin
        try:
            plt.plot(trajectory_m[:, 0], trajectory_m[:, 1], label=f"Rover {rover.id}")
        except:
            print(f"Rover {rover.id} has no trajectory data.")
            rover.finished = True

        # Plot timed goals
        goals = np.array([(g[1], g[2]) for g in rover.timed_goals])
        if len(goals) > 0:
            goals_m = goals * grd_size * resolution + map_origin
            plt.scatter(goals_m[:, 0], goals_m[:, 1], marker='x', s=50)
            # plt.scatter(goals_m[:, 0], goals_m[:, 1], marker='x', s=50, label=f"Goals {rover.id}")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Rover Trajectories and goals on Lunar Map")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(output_dir, "trajectories/")
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + 'trajectories_plot.png', dpi=500)
    plt.close()

def plot_final_trajectories_contrast(surface, output_dir='.'):
    print('Plotting rovers trajectories...')
    # Map data
    obstacle_map = (surface.ground >= 100).astype(np.uint8)

    # Origin and extent
    origin = (map_origin, map_origin)
    grd_size = surface.grd_size
    height = grd_size * resolution
    extent = [
        origin[0],
        origin[0] + grd_size * resolution,
        origin[1],
        origin[1] + height
    ]

    # High-contrast color cycle (ColorBrewer Set1)
    color_cycle = [
        '#e41a1c',  # red
        '#377eb8',  # blue
        '#4daf4a',  # green
        '#984ea3',  # purple
        '#ff7f00',  # orange
        '#ffff33',  # yellow
        '#a65628',  # brown
        '#f781bf',  # pink
        '#999999'   # gray
    ]

    plt.figure(figsize=(10, 10))
    plt.imshow(obstacle_map, cmap="gray_r", origin="lower", extent=extent)

    for i, rover in enumerate(surface.rovers):
        color = color_cycle[i % len(color_cycle)]
        trajectory = np.array(rover.trajectory)
        trajectory_m = trajectory * grd_size * resolution + map_origin

        try:
            plt.plot(trajectory_m[:, 0], trajectory_m[:, 1],
                     label=f"Rover {rover.id}", color=color, linewidth=3)
        except:
            print(f"Rover {rover.id} has no trajectory data.")
            rover.finished = True

        # Plot timed goals
        goals = np.array([(g[1], g[2]) for g in rover.timed_goals])
        if len(goals) > 0:
            goals_m = goals * grd_size * resolution + map_origin
            plt.scatter(goals_m[:, 0], goals_m[:, 1], marker='x', s=80, color=color)
            

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Rover Trajectories and Goals on Lunar Map")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "map/trajectories_plot.png"), dpi=500)
    plt.close()

# =============================================================================
# Rover exploration
# =============================================================================

def divide_map(surface, output_dir=outputPath):
    """
    Divide the unexplored map among rovers using K-means clustering and Hungarian assignment.
    - Clusters unexplored, rock-free cells into num_rovers regions.
    - Assigns each region to the closest rover using the Hungarian algorithm.
    - Each rover stores its assigned area (list of (row, col) grid cells) in rover.assigned_area.
    - Plots the resulting region assignments and rover positions.
    - Saves the plot to 'map/map_division.png' in the output directory.
    """
    print(f'Dividing map into {num_rovers} exploration areas')

    unexplored = np.argwhere((surface.ground == 0) & (surface.rocks == 0))
    if len(unexplored) < num_rovers:
        print("Not enough unexplored cells to assign to all rovers.")
        return None

    # 1. K-means clustering of unexplored space
    kmeans = KMeans(n_clusters=num_rovers, n_init=20, random_state=42)
    labels = kmeans.fit_predict(unexplored)
    cluster_centers = kmeans.cluster_centers_

    # 2. Hungarian assignment to rovers
    rover_grid_pos = np.array([surface.pos_to_cell(rover.position) for rover in surface.rovers])
    dist_matrix = np.linalg.norm(rover_grid_pos[:, None, :] - cluster_centers[None, :, :], axis=2)
    rover_indices, cluster_indices = linear_sum_assignment(dist_matrix)
    assignments = {cluster: rover for cluster, rover in zip(cluster_indices, rover_indices)}

    # 3. Region assignment map and per-rover assigned area
    region_map = np.full_like(surface.ground, fill_value=-1, dtype=int)
    # Clear previous assignments
    for rover in surface.rovers:
        rover.assigned_area = []
    for idx, cell in enumerate(unexplored):
        cluster = labels[idx]
        rover_id = assignments[cluster]
        region_map[cell[0], cell[1]] = rover_id
        # Store assigned area in rover
        surface.rovers[rover_id].assigned_area.append(tuple(cell))

    # Attach a per-rover boolean region mask
    for i, rover in enumerate(surface.rovers):
        rover.region_map = (region_map == i)

    # 4. Plot with same style as trajectory map
    plt.figure(figsize=(10, 10))

    # Obstacle base layer (rocks = black)
    obstacle_map = (surface.ground >= 100).astype(np.uint8)
    origin = (map_origin, map_origin)
    grd_size = surface.grd_size
    height = grd_size * resolution
    extent = [
        origin[0],
        origin[0] + grd_size * resolution,
        origin[1],
        origin[1] + height
    ]
    plt.imshow(obstacle_map, cmap="gray_r", origin="lower", extent=extent)

    # Region overlay (transparent)
    # cmap = plt.get_cmap("tab10", num_rovers)

    # Use matplotlib's default property cycle to match plot_final_trajectories
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    cmap = ListedColormap(colors[:num_rovers])
    region_img = np.ma.masked_where(region_map == -1, region_map)
    plt.imshow(region_img, cmap=cmap, origin="lower", extent=extent, alpha=0.5)

    # Rover positions
    # for i, rover in enumerate(surface.rovers):
    #     r, c = surface.pos_to_cell(rover.position)
    #     x = c * resolution + map_origin
    #     y = r * resolution + map_origin
    #     plt.plot(x, y, 'o', color=cmap(i), label=f"Rover {i+1}", markersize=10)

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Exploration Region Assignment")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.join(output_dir, "trajectories"), exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"trajectories/map_division_{num_rovers}.png"), dpi=500)
    plt.close()

def run_exploration(surface, obs_radius=frontier_radius, output_dir='.'):
    """
    1) divide_map, fills rover.assigned_area
    2) reset all rovers back to the lander
    3) build an initial A* path for each rover into its assigned region
    4) maintain per-rover known-map with radius obs_radius
    5) loop until all rovers finished:
         - if rover has no current path: detect frontiers, pick nearest, log timed_goal, plan full A* path
         - else pop one cell, mark known patch, append trajectory
    6) save trajectories.csv & goals.csv into output_dir/trajectories
    7) Resets rover parameters
    8) Sets the simulation time to min(sim_time, max(rover trajectories length)). This is done in order to truncate the simulation when rovers are not moving anymore.

    Each step, a rover moves from one grid cell to an adjacent cell along its A* path.
    The physical distance per step is (resolution) meters (e.g., 0.05 m if resolution=0.05).
    """
    grd_size = surface.grd_size

    # 1) partition the map among rovers
    divide_map(surface, output_dir)

    # 2) reset all rovers to the lander starting position
    for r in surface.rovers:
        r.position   = surface.lander.copy()
        r.trajectory = [tuple(r.position)]
        r.timed_goals = []
        r.finished   = False

    # 3) plan an initial A* path into each rover's region
    for r in surface.rovers:
        start = surface.pos_to_cell(r.position)
        # pick the closest cell in their assigned area
        target = min(r.assigned_area,
                     key=lambda c: np.linalg.norm(np.array(c) - np.array(start)))
        # record that move as a timed goal at t=0
        gx = (target[1] + 0.5) / grd_size
        gy = (target[0] + 0.5) / grd_size
        r.timed_goals.append((0, gx, gy))
        # compute the full A* path into the region
        path = r.astar(surface.rocks, start, target)
        r._path_cells = path[1:] if len(path) > 1 else []

    # 4) initialize per-rover known maps
    known = {}
    for r in surface.rovers:
        km = np.zeros((grd_size, grd_size), bool)
        rr, cc = surface.pos_to_cell(r.position)
        for dr in range(-obs_radius, obs_radius + 1):
            for dc in range(-obs_radius, obs_radius + 1):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < grd_size and 0 <= nc < grd_size and dr*dr + dc*dc <= obs_radius*obs_radius:
                    km[nr, nc] = True
        known[r.id] = km

    # 5) exploration loop until everyone is done
    t = 0
    while True:
        any_active = False
        for r in surface.rovers:
            if r.finished:
                continue
            any_active = True

            # if no path, pick a new frontier in their region
            if not r._path_cells:
                frs = []
                mask = known[r.id]
                for (rr, cc) in r.assigned_area:
                    if mask[rr, cc]:
                        continue
                    # frontier if adjacent to known
                    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                        nr, nc = rr+dr, cc+dc
                        if 0 <= nr < grd_size and 0 <= nc < grd_size and mask[nr, nc]:
                            frs.append((rr, cc))
                            break
                if not frs:
                    r.finished = True
                    continue
                # choose nearest frontier
                curr = surface.pos_to_cell(r.trajectory[-1])
                arr = np.array(frs)
                dists = np.linalg.norm(arr - np.array(curr), axis=1)
                goal = frs[int(np.argmin(dists))]
                # log timed goal
                gx = (goal[1] + 0.5) / grd_size
                gy = (goal[0] + 0.5) / grd_size
                r.timed_goals.append((t, gx, gy))
                # plan full A* path to it
                full = r.astar(surface.rocks, curr, goal)
                r._path_cells = full[1:] if len(full) > 1 else []

            # if path exists, take one step along it
            if r._path_cells:
                nr, nc = r._path_cells.pop(0)
                # mark observation radius around new cell
                mask = known[r.id]
                for dr in range(-obs_radius, obs_radius + 1):
                    for dc in range(-obs_radius, obs_radius + 1):
                        ar, ac = nr+dr, nc+dc
                        if 0 <= ar < grd_size and 0 <= ac < grd_size and dr*dr + dc*dc <= obs_radius*obs_radius:
                            mask[ar, ac] = True
                # append normalized position
                x_n = (nc + 0.5) / grd_size
                y_n = (nr + 0.5) / grd_size
                r.trajectory.append((x_n, y_n))

        # stop when all rovers are finished
        if not any_active:
            break
        t += 1

    # 6) save CSVs and 7) plot results
    traj_dir = os.path.join(output_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    surface.export_rover_trajectories(traj_dir)
    surface.export_rover_goals(traj_dir)

    # 7) reset rovers.finished to make exploration
    for r in surface.rovers:
        if len(r.trajectory) > 0:
            r.finished = False
        
        r.timed_goals_index = 0
    
    # 8) set the simulation time to min(sim_time, max(rover trajectories length))
    global sim_duration, frames_count
    max_traj_len = max(len(r.trajectory) for r in surface.rovers)
    if max_traj_len < sim_duration:
        print(f"Truncating sim_duration from {sim_duration} to {max_traj_len} to match rover trajectories.")
        sim_duration = max_traj_len
        frames_count = sim_duration // time_interval

# =============================================================================
# Learning
# =============================================================================

class ExperienceReplay:
    def __init__(self, maxlen=10000):
        self.buffer = deque(maxlen=maxlen)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class RoverDDQNAgent:
    def __init__(self, surface, state_size=state_size_node*(max_neighbors+1), action_size=max_neighbors+1, gamma=0.99, epsilon=0.1,
                 epsilon_min=0.01, epsilon_decay=0.9995, buffer_size=256, batch_size=32, ddqn=True, action_seed=action_seed):
        self.surface = surface
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilons = []
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.ddqn = ddqn
        self.replay = ExperienceReplay(maxlen=buffer_size)
        self.train_step = 0
        self.act_seed=np.random.default_rng(action_seed)

    def buffer_penalty(self, node, alpha=buffer_steppness):
        '''
        Exponential penalty [0, penalty_drop] vs. buffer fill (alpha=steepness).
        '''
        usage_ratio = node.get_buffer_use() / buffer_size
        # return penalty_drop * (np.exp(alpha * usage_ratio) - 1) / (np.exp(alpha) - 1)
        return penalty_drop * (10**(alpha * usage_ratio) - 1) / (10**alpha - 1)
   
    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        # model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        # model.compile(loss='mse', optimizer='adam')
        optimizer = Adam(clipnorm=1.0)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def get_action_index(self, state):
        '''
        Exploration vs exploitation poicy implemented. If the agent tries to select an invalid action,
        meaning an unavailable interface, a negative reward will be given and the agent will try another action
        based on the current policy.
        '''
        tried_actions = set()
        q_values = None
        self.epsilons.append((self.epsilon, self.surface.sim_time))
        while True:
            if not explore_train:
                self.epsilon = 0
            if self.act_seed.random() <= self.epsilon:
                action = random.randrange(self.action_size)
            else:
                if q_values is None:
                    # state = np.asarray(state, dtype=np.float32)
                    q_values = self.q_network.predict(state[np.newaxis], verbose=0)[0]
                    if np.isnan(q_values).any():
                        print(f"NaN in Q-values: {q_values}")

                action = np.argmax(q_values)

            # Ensure the action is not already tried
            if action in tried_actions:
                continue

            # Check if the action is valid
            if action == 0:  # Action 0 corresponds to 'hold', which is always valid
                return action
            neighbor_index = (action - 1) * state_size_node + state_size_node  # Calculate the starting index for the neighbor in the state
            if neighbor_index < len(state) and state[neighbor_index] != unavailable:
                return action  # Valid action

            # If invalid, store a bad reward experience and retry. # FIXME this should not happen in exploitation if a model is well trained
            self.store_experience(state, action, penalty_unavailable, state, False)
            tried_actions.add(action)  # Mark this action as tried

            # Force the invalid action's Q-value to -inf
            if q_values is not None:
                q_values[action] = -float('inf')

            # If all actions have been tried, break to avoid infinite loop
            if len(tried_actions) == self.action_size:
                break

    def store_experience(self, state, action, reward, next_state, done):
        self.replay.store(state, action, reward, next_state, done)

    def train(self):
        if len(self.replay) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        q_targets = self.q_network.predict(states, verbose=0)
        if self.ddqn:
            q_next = self.q_network.predict(next_states, verbose=0)
            q_target_next = self.q_target.predict(next_states, verbose=0)
            best_actions = np.argmax(q_next, axis=1)
            target_vals = rewards + self.gamma * q_target_next[np.arange(self.batch_size), best_actions] * (1 - dones)
        else:
            q_next = self.q_target.predict(next_states, verbose=0)
            target_vals = rewards + self.gamma * np.max(q_next, axis=1) * (1 - dones)

        q_targets[np.arange(self.batch_size), actions] = target_vals
        self.q_network.fit(states, q_targets, epochs=1, verbose=0, batch_size=self.batch_size)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.train_step += 1
        if self.train_step % update_f == 0:
            self.update_target_network()

    def update_target_network(self):
        self.q_target.set_weights(self.q_network.get_weights())

    def observe_state(self, node):
        """
        The passed node is either a rover or the lander.
        Builds the local observation vector:
        [x_i, y_i, connected_i, distance_i, ttl_i, x_obj_i, y_obj_i,
        x_n1, y_n1, connected_n1, distance_n1, ttl_n1, x_obj_n1, y_obj_n1,
        ..., (padded with -1 if neighbor missing)]
        All distance values scaled to [0,10] where applicable.
        TTL and connected use {0, 10}.
        """
        state = []

        # --- Current node state ---        
        state += [node.buffer_normalized,       # Normalized buffer use,
                  node.connected,               # Connected to lander,
                  node.ttl]                     # Time steps to get to lander (TTL) coverage area on the way for next objective,
        if HTL:
            state += [node.htl]                 # Hops to lander    

        # --- Neighbor states ---
        sorted_neighbors = sorted(
            [node.surface.get_node_by_id(nid) for nid in node.local_neighbors], key=lambda neighbor: neighbor.htl)[:max_neighbors]
        
        for i in range(max_neighbors):
            if i < len(sorted_neighbors):
                state += [sorted_neighbors[i].buffer_normalized, sorted_neighbors[i].connected, sorted_neighbors[i].ttl]
                if HTL:
                    state += [sorted_neighbors[i].htl]
            else:
                state += [unavailable] * state_size_node

        return np.array(state, dtype=np.float32), sorted_neighbors
    

    # def observe_state_v0(self, node):
    #     """
    #     The passed node is either a rover or the lander.
    #     Builds the local observation vector:
    #     [x_i, y_i, connected_i, distance_i, ttl_i, x_obj_i, y_obj_i,
    #     x_n1, y_n1, connected_n1, distance_n1, ttl_n1, x_obj_n1, y_obj_n1,
    #     ..., (padded with -1 if neighbor missing)]
    #     All distance values scaled to [0,10] where applicable.
    #     TTL and connected use {0, 10}.
    #     """
    #     state = []

    #     # --- Current node state ---        
    #     state += [node.position[0] * pos_norm,  # [Current node x normalized position, 
    #               node.position[1] * pos_norm,  # Current node y normalized position, 
    #               node.buffer_normalized,       # Normalized buffer use,
    #               node.connected,               # Connected to lander,
    #               node.distance_to_lander,      # Distance to lander,
    #               node.ttl,                     # Time to lander (TTL) for next objective,
    #               node.objective[0] * pos_norm, # x position of the objective normalized,
    #               node.objective[1]] * pos_norm # y position of the objective normalized]

    #     # --- Neighbor states ---
    #     sorted_neighbors = sorted(node.local_neighbors)[:max_neighbors]
    #     for i in range(max_neighbors):
    #         if i < len(sorted_neighbors):
    #             neighbor = node.surface.get_node_by_id(sorted_neighbors[i])
    #             state += [neighbor.position[0] * pos_norm, neighbor.position[1] * pos_norm, neighbor.buffer_normalized, neighbor.connected, neighbor.distance_to_lander, neighbor.ttl, neighbor.objective[0] * pos_norm, neighbor.objective[1] * pos_norm]
    #         else:
    #             state += [-1] * state_size_node

    #     return np.array(np.array(state, dtype=np.float32))

# =============================================================================
# Packet Class
# =============================================================================
class Packet:
    """
    Class representing a data packet (a map) generated by a rover.
    Attributes:
      - id: Unique packet identifier (e.g. '1_7_0' (roverID_packetCounter_packetCopyCounter)).
      - source_id: ID of the rover that generated the packet.
      - origin: Coordinates where the map is from.
      - size: Size of the packet in bits.
      - delivered_to_lander_time: Time when first delivered (None if not delivered).
      - dropped_time: Time when dropped (None if not dropped).
    """
    def __init__(self, packet_id, source_id, origin, size, creation_time, objective=None, L=None):
        self.id = packet_id
        self.source_id = source_id
        self.origin = np.array(origin)
        self.size = size
        self.creation_time = creation_time
        self.delivered_to_lander_time = None
        self.dropped_time = None
        self.path = [[source_id, creation_time]]
        self.objective = objective
        self.L = L
        self.copy_counter = L           # remaining number of copies that can be done to this packet for Spray and Wait policy

# =============================================================================
# Lander Class
# =============================================================================
class Lander:
    """
    Class representing the lander.
    It stores received packets in an (effectively) infinite buffer.
    """
    def __init__(self, position, surface):
        self.id = 0
        self.position = np.array(position)
        self.buffer = []            # List to store received Packet objects
        self.processed_buffer = []  # List of packets that have already been processed
        self.duplicatedBuffer = []  # List of received duplicated packets
        self.surface = surface
        
        # Learning
        self.update_local_graph_lander()
        # self.ttl = ttl_max
        self.ttl = 0
        self.objective = self.position
        self.connected = 0
        self.distance_to_lander = 0
        self.buffer_normalized = 0
        self.max_q_value = reward_deliver
        self.htl = 0
        self.sorted_neighbors = []  # sorted list of neighbors based on hops to lander. In this case the sort will be arbitrary since all of them are connected to the lander

    def receive_packet(self, packet, current_time):
        """
        Append a received packet to the buffer.
        If this is the first delivery, record the delivery time.
        """
        packet.path.append(['Lander', current_time])
        if packet.delivered_to_lander_time is None:
            packet.delivered_to_lander_time = current_time
        self.buffer.append(packet)

    def update_local_graph_lander(self):
        self.update_local_subgraph_lander()
        self.update_direct_neighbors_lander()

    def update_local_subgraph_lander(self):
        nodes = nx.node_connected_component(self.surface.G_feasible, self.id)
        self.G_local = self.surface.G_feasible.subgraph(nodes)

    def update_direct_neighbors_lander(self):
        self.local_neighbors = list(self.G_local.neighbors(self.id))

# =============================================================================
# Rover Class
# =============================================================================
class Rover:
    def __init__(self, rover_id, initial_position, surface, s_color='blue', L=None):
        """
        Initialize a rover with a given ID, starting position, and reference to the LunarSurface.
        Also initializes local buffers, outgoing queues, a packet counter, and a color.
        """
        self.id = rover_id
        self.position = np.array(initial_position)
        self.surface = surface
        self.objective = None       # next rover's objective to be explored
        self.objectives_list = []   # list of (x, y) objectives
        self.objectives_index = 0   # track which objective is next
        self.trajectory = []        # stores rover trajectory points
        self.path = []              # Precomputed A* path (list of grid cells)
        self.timed_goals = []       # List of the goals followed by the rover
        self.finished = False       # True when rover has finished exploring
        self.active_buffer = []     # Local FIFO buffer for Packet objects
        self.inactive_buffer = []   # Local FIFO buffer for Packet objects
        self.dropped_packets = []   # List for dropped Packet objects
        self.outgoing_queues = {}   # Mapping neighbor_id -> list (queue) of Packet objects
        self.packet_counter = 0     # To assign unique packet IDs
        self.color = load_cmap(0)   # Color based solely on local buffer length
        self.static_color = s_color # Set rover's static color
        self.tx_count = 0           # Count of packets transmitted in current step
        self.step = 0               # Step counter
        self.objectives_missed = [] # list of missed objectives because the buffer was full
        self.objective_counter = 0  # number of reached objectives
        self.wait_counter = wait    # rovers will wait this time steps after reaching an objective
        self.G_local = None         # local rover direct connected graph
        self.local_neighbors = []   # list of the local neighbors connected through a direct link
        self.L = L                  # Max number of copies per packet. Parameter for flooding algorithms like Spray and wait
        self.buffer_use = []        # keeps track of the historic buffer use and congestion
        self.compute_ttl()          # computes Time To Lander in self.ttl. changes every time a new objective is set
        self.connecteds = []        # list of tuples (connected, time) to keep track of the connectedness of the rover to the lander
        self.update_connected()     # computes the connectedness of the rover to the lander
        self.current_state = None   # current state
        self.last_state = None      # last state observed
        self.last_action = None     # last action taken
        self.last_action_destination = None # Which node id did last action made the packet go to
        self.dropped_packet = False # True if the rover dropped packets in the last step
        self.blocked_packet = False # True if the rover was blocked in the last step because it was full
        self.rewards = []           # list of received rewards
        self.buffer_normalized = 0  # normalized buffer size
        # self.update_QValue()        # Given a current stat, updates the highest possible Q-Value of the rover
        self.htl = 0                # Initializes Hop To Lander
        self.sorted_neighbors = []  # sorted list of neighbors based on hops to lander
        self.copies = 0             # number of copies of the packet that are in the rover. Only used for flooding algorithms like Spray and Wait

    def update_hops_to_lander(self):
        '''
        Updates the hops to lander (htl) based on the rover's reachable graph (G_local).
        - If directly connected to the lander, htl = 0.
        - If indirectly connected, htl is the number of hops in the shortest path.
        - If not connected (directly or indirectly), htl = unconnected (50).
        '''
        if self.connected == connected:
            self.htl = 1
            return

        # Compute shortest path length to the lander in the local graph
        if 0 in self.G_local:
            self.htl = nx.shortest_path_length(self.G_local, source=self.id, target=0)
        else:
            # If the lander is not in the local graph, set htl to unconnected
            self.htl = unconnected

    # def update_hops_to_lander_v0(self):
    #     '''
    #     From the last state, gets the smallest number of hops to the lander and adds one.
    #     '''
    #     if self.connected == connected:
    #         self.htl = 0
    #         return
    #     if self.prev_connected == True and self.connected == unconnected:
    #         # If the rover just disconnected, set hops to lander to max to reinitialize counter
    #         self.htl = hops_to_lander
    #         return
    #     elif self.current_state is not None:
    #         # Extract hops to lander for neighbors from self.current_state
    #         neighbor_hops = [
    #             self.current_state[state_size_node * (i + 1) + 3]
    #             for i in range(max_neighbors)
    #             if self.current_state[state_size_node * (i + 1) + 3] not in (unavailable, -1)
    #         ]
    #         # Find the smallest hops to lander among neighbors
    #         smallest_hops = min(neighbor_hops, default=hops_to_lander)
    #         if smallest_hops != hops_to_lander:
    #             self.htl = smallest_hops + 1
    #     else:
    #         # If current state is None, set hops to lander to max
    #         self.htl = hops_to_lander

    def update_connected(self):
        """
        Update the connectedness of the rover to the lander.
        If the rover is connected to the lander (line of sight and within local neighborhood),
        set self.connected to connected. Otherwise, set it to unconnected.
        Also updates the distance to the lander in self.distance_to_lander.
        """
        if hasattr(self, 'connected'):
            self.prev_connected = self.connected==connected
        else:
            self.prev_connected = False
        self.distance_to_lander = np.linalg.norm(self.position - self.surface.lander) * distance_norm
        if 0 in self.local_neighbors and self.surface.line_of_sight_clear(self.position, self.surface.lander):
            self.connected = connected
        else:
            self.connected = unconnected
        self.connecteds.append((self.connected, self.surface.sim_time))

    def in_lander_coverage(self, pos):
        """
        Returns True if 'pos' is inside the lander's communication coverage area.
        """
        return (np.linalg.norm(pos - self.lander) < threshold_base) and self.line_of_sight_clear(pos, self.lander)
    
    def compute_ttl(self, objectives_ahead=2):
        """
        Computes the number of steps needed for the rover to reach a position
        where it has direct line-of-sight to the lander *and* is within threshold distance.
        """
        if traj_type > 0 and self.trajectory:
            ttl_counter = 0
            objectives_found = 0
            current_step = self.step

            # Check if there are timed goals defined
            goals_steps = [goal[0] for goal in self.timed_goals[self.timed_goals_index:]]
            goals_steps_set = set(goals_steps)

            while current_step < len(self.trajectory):
                rover_pos = np.array(self.trajectory[current_step])

                if self.surface.line_of_sight_clear(rover_pos, self.surface.lander):
                    if np.linalg.norm(rover_pos - self.surface.lander) < threshold_base:
                        self.ttl = ttl_counter / ttl_norm
                        return

                if current_step in goals_steps_set:
                    objectives_found += 1
                    if objectives_found >= objectives_ahead:
                        break

                ttl_counter += 1
                current_step += 1

            self.ttl = no_ttl / ttl_norm  # If no coverage found
            return

        # A* paths (traj_type=0)
        if not self.path:
            self.ttl = no_ttl / ttl_norm
            return

        ttl_counter = 0
        if not self.prev_connected and self.connected == connected:
            self.ttl = 1 / ttl_norm
            return

        elif self.connected == connected:
            self.ttl = ttl_counter
            return

        for cell in self.path:
            rover_x = (cell[1] + 0.5) / self.surface.grd_size
            rover_y = (cell[0] + 0.5) / self.surface.grd_size
            rover_pos = np.array([rover_x, rover_y])
            if self.surface.line_of_sight_clear(rover_pos, self.surface.lander):
                if np.linalg.norm(rover_pos - self.surface.lander) < threshold_base:
                    self.ttl = ttl_counter / ttl_norm
                    return
            ttl_counter += 1

        self.ttl = no_ttl / ttl_norm

    def make_action(self, action_idx, neighbors):
        """
        Executes the chosen action on packets:
        - 0 = hold: Keep packets in the current rover's buffer.
        - 1..len(neighbors) = forward to neighbors[action_idx-1]: Forward packets to the selected neighbor.
        Sends as many packets as possible to the chosen destination, limited by tx_rate_indirect or tx_rate_direct.
        """
        # HOLD
        if action_idx == 0:
            # No action needed, packets remain in the current rover's buffer
            return

        # Map action index to neighbor ID
        nbr_i = action_idx - 1
        if nbr_i < 0 or nbr_i >= len(neighbors):
            # Invalid index → treat as hold
            print(f'invalid action index: {action_idx} for rover {self.id}')
            return

        dest_id = neighbors[nbr_i].id

        # DELIVER TO LANDER
        if dest_id == 0:
            # Deliver packets directly to the lander, limited by tx_rate_direct
            num_to_send = tx_rate_direct
            sent_to_lander = 0
            while self.active_buffer and num_to_send > 0:
                packet = self.active_buffer.pop(0)
                self.surface.lander_obj.receive_packet(packet, self.surface.sim_time)
                num_to_send -= 1
                sent_to_lander += 1
            # Record link usage
            if sent_to_lander > 0:
                self.surface.link_usage[(self.id, 0)] = self.surface.link_usage.get((self.id, 0), 0) + sent_to_lander
            return

        # FORWARD TO ANOTHER ROVER
        next_rover = self.surface.get_node_by_id(dest_id)
        num_to_send = tx_rate_indirect
        sent_to_rover = 0
        while self.active_buffer and num_to_send > 0 and not next_rover.is_buffer_full():
            packet = self.active_buffer.pop(0)
            next_rover.receive_packet(packet)
            num_to_send -= 1
            sent_to_rover += 1
        # Record link usage
        if sent_to_rover > 0:
            self.surface.link_usage[(self.id, dest_id)] = self.surface.link_usage.get((self.id, dest_id), 0) + sent_to_rover

    def learning_step(self):
        """
        DQN based packet handling under comm_policy==2:
          - process up to capacity packets,
          - pick action, log reward for previous action,
          - execute via make_action(),
          - invalid actions are just holds.
        """
        neighbors = self.sorted_neighbors
        
        # Surface agent
        agent = self.surface.DDQNAgent

        #--------------------------------------------
        # process last experience
        #--------------------------------------------
        if explore_train or not import_models:
            if self.last_state is not None and self.last_action is not None:
                
                # Last time holded packets
                if self.last_action_destination.id == self.id:
                    next_state = self.current_state
                    reward = agent.buffer_penalty(self) + penalty_hold
                
                # Last time delivered packets
                elif self.last_action_destination.id == 0:
                    next_state = self.last_action_destination.current_state
                    reward = reward_deliver
                
                # Last time forwarded packets
                else:
                    next_state = self.last_action_destination.current_state
                    reward = agent.buffer_penalty(self.last_action_destination) + penalty_forward   # penalty for buffer use and for forwarding

                # Store experience
                agent.store_experience(self.last_state, self.last_action, reward, next_state, done=False)
                self.rewards.append((reward, self.surface.sim_time))

        #--------------------------------------------
        # Choose and perform next action
        #--------------------------------------------
  
        # Choose action
        action_index = agent.get_action_index(self.current_state)
        next_node = self if action_index==0 else neighbors[action_index - 1]  # Get the neighbor object or self for hold action
        self.make_action(action_index, neighbors)

        # Train the agent
        if explore_train:
            agent.train()

        # store last action
        self.last_action = action_index
        self.last_action_destination = next_node

    def drop_first(self):
        """
        Drop the first packet available, prioritizing inactive_buffer.
        If inactive_buffer is empty, try active_buffer.
        Return True if a packet was dropped, else False.
        """
        if self.inactive_buffer:
            dropped_packet = self.inactive_buffer.pop(0)
        elif self.active_buffer:
            dropped_packet = self.active_buffer.pop(0)
        else:
            return False  # Nothing to drop

        dropped_packet.dropped_time = self.surface.sim_time
        self.dropped_packets.append(dropped_packet)
        return True

    def generate_objective(self):
        """
        Generate a random exploration objective (a point in [0,1]^2) not on a rock.
        Uses the surface's objective_rng for reproducibility.
        """
        valid = False
        while not valid:
            candidate = self.surface.objective_rng.random(2)
            cell = self.surface.pos_to_cell(candidate)
            if self.surface.rocks[cell] == 0:
                valid = True
                self.objective = candidate
        self.timed_goals.append((self.step, candidate[0], candidate[1]))
        return self.objective

    @staticmethod
    def astar(grid, start, goal):
        """
        Perform A* search on a grid.
        grid: 2D numpy array; cells with value 1 are obstacles.
        start, goal: (row, col) tuples.
        Uses Manhattan distance as the heuristic.
        Returns a list of grid cells from start to goal (inclusive), or [] if no path.
        """
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        def heuristic(a, b):
            return abs(a[0]-b[0]) + abs(a[1]-b[1])
        open_set = []
        heapq.heappush(open_set, (heuristic(start, goal), start))
        came_from = {}
        gscore = {start: 0}
        while open_set:
            current_f, current = heapq.heappop(open_set)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            for d in neighbors:
                neighbor = (current[0] + d[0], current[1] + d[1])
                if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]):
                    continue
                if grid[neighbor] == 1:
                    continue
                tentative_g = gscore[current] + 1
                if tentative_g < gscore.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g
                    fscore = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (fscore, neighbor))
        return []  # no path found

    def plan_path(self):
        """
        Compute the A* path from the rover's current position to its objective.
        The entire path is generated at once.
        """
        start_cell = self.surface.pos_to_cell(self.position)
        goal_cell = self.surface.pos_to_cell(self.objective)
        new_path = Rover.astar(self.surface.rocks, start_cell, goal_cell)
        self.path = new_path[1:] if len(new_path) > 1 else []

    def update_position(self, step_size):
        """
        Move along the precomputed A* path toward the objective.
        Adds random noise, generates one new packet every time step (and an extra upon reaching target).
        Updates the rover's color based solely on its local buffer length.
        """
        # Random trajectories
        if not import_trajectories and traj_type==0:
            # wait some time
            if self.wait_counter > 0:
                self.wait_counter -= 1
                return
            if not self.path:   # objective reached
                # Generate packets_per_objective additional packets when the objective is reached
                for i in range(packets_per_objective):
                    self.generate_packet(packet_size, self.objective_counter)
                # Generate new objective
                self.generate_objective()
                self.plan_path()
                self.wait_counter = wait

            if self.path:
                next_cell = self.path[0]
                target_x = (next_cell[1] + 0.5) / self.surface.grd_size
                target_y = (next_cell[0] + 0.5) / self.surface.grd_size
                target_pos = np.array([target_x, target_y])
                direction = target_pos - self.position
                dist = np.linalg.norm(direction)
                if noise_movement:
                    # noise = noise_strength * step_size* np.random.randn(2)
                    noise = noise_strength * step_size* self.surface.movement_rng.random(2)
                else:
                    noise = 0
                movement = (step_size * direction/dist) if dist != 0 else 0
                candidate = self.position + movement + noise
                cell = self.surface.pos_to_cell(candidate)
                if self.surface.rocks[cell] == 1:
                    return  # Skip movement if candidate lands on a rock.
                self.position = candidate
                if np.linalg.norm(target_pos - self.position) < step_size:
                    self.position = target_pos
                    self.path.pop(0)
                if self.step % packet_generation_flow == 0:
                    self.generate_packet(packet_size)
                self.step += 1
        
        # trajectories generated by PSE
        elif traj_type > 0:
            # 1) Follow the main (predefined) trajectory
            if not self.finished:
                if self.step < len(self.trajectory):
                    next_pos = self.trajectory[self.step]
                    # Check for NaNs to detect actual end
                    if np.isnan(next_pos[0]) or np.isnan(next_pos[1]):
                        self.finished = True
                        # self.position = self.trajectory[self.step - 1]  
                        self.position = np.array(self.trajectory[self.step - 1])    # last valid
                        self.trajectory.append(tuple(self.position))                # append last pos
                        print(f"Rover {self.id} finished exploring at step {self.step}.")
                        surface.check_finished_rovers()
                        return
                    
                    self.position = np.array(self.trajectory[self.step])
                    self.step += 1
                    
                    # Generate a packet every 'packet_generation_flow' steps
                    if self.step % packet_generation_flow == 0:
                        self.generate_packet(packet_size)
                else:
                    self.finished = True
                    print(f"Rover {self.id} finished exploring at step {self.step}.")
                    surface.check_finished_rovers()
                    return
                    
                # 2) Check if we have a timed goal that matches or is behind our current step
                if hasattr(self, 'timed_goals') and self.timed_goals_index < len(self.timed_goals):
                    # if Docker's time is exactly self.step
                    docker_time, gx, gy = self.timed_goals[self.timed_goals_index]
                    if self.step >= docker_time:
                        # We reached or passed that Docker time => treat that as a new objective
                        self.objective = (gx, gy)
                        # Generate 'packets_per_objective' upon each new goal
                        for _ in range(packets_per_objective):
                            self.generate_packet(packet_size, self.timed_goals_index)
                        self.timed_goals_index += 1
        
        # Record historic
        self.buffer_use.append((self.get_buffer_use(), self.step))
        if traj_type == 0:
            self.trajectory.append(tuple(self.position))

        # Update color based solely on local buffer length.
        self.color = load_cmap(self.get_buffer_use() / buffer_size)

    def generate_packet(self, size, objective = None):
        """
        Create a new Packet from the rover's current data.
        The packet ID is 'roverID_packetCounter_packetCopyCounter'.
        If the local buffer already contains buffer_size packets, drop the packet (recording drop time).
        """
        if not generate_traffic:
            return # traffic generation is disabled
        
        packet_id = f"{self.id}_{self.packet_counter}"
        self.packet_counter += 1
        packet = Packet(packet_id, self.id, self.position, size, self.surface.sim_time, objective, L=self.L)
        if (self.get_buffer_use()) >= buffer_size:
            self.drop_first()
            self.dropped_packet = True
        else:
            self.dropped_packet = False
        self.active_buffer.append(packet)

    def update_local_subgraph(self):
        # Compute the connected component containing this rover, then build the subgraph.
        nodes = nx.node_connected_component(self.surface.G_feasible, self.id)
        self.G_local = self.surface.G_feasible.subgraph(nodes)
    
    def update_direct_neighbors(self):
        # Returns a list of nodes that are directly connected to this rover in the local subgraph.
        self.local_neighbors = list(self.G_local.neighbors(self.id))

    def update_queues(self):
        """
        Update the outgoing queues using the local feasible subgraph.
        The local component is computed from G_feasible, and the transmission
        capacity is set to: capacity = (number of nodes in local component) * tx_rate_indirect.
        For each of the first 'capacity' packets in self.active_buffer, assign it to the neighbor 
        (from the local component) with the minimum hop count (using G_feasible) to the lander.
        """
        
        capacity = len(self.local_neighbors) * tx_rate_indirect
        # Exclude self from neighbors.
        neighbors = self.local_neighbors
        queue_assignments = {}  # neighbor_id -> list of packets
        # For each packet among the first 'capacity' packets in the buffer:
        for packet in self.active_buffer[:capacity]:
            best_neighbor = None
            min_hops = float('inf')
            for n in neighbors:
                # --- Skip neighbors that are completely full ---
                if self.surface.get_node_by_id(n).is_buffer_full():
                    continue
                try:
                    hops = nx.shortest_path_length(self.G_local, source=n, target=0)
                except nx.NetworkXNoPath:
                    hops = float('inf')
                if hops < min_hops:
                    min_hops = hops
                    best_neighbor = n
            if best_neighbor is not None:
                queue_assignments.setdefault(best_neighbor, []).append(packet)
        self.outgoing_queues = queue_assignments

    def process_queues(self):
        """
        Process (transmit) packets from each outgoing queue.
        For each neighbor's queue, up to tx_rate_indirect packets are transmitted.
        Each transmitted packet is removed from both the queue and from the local buffer.
        Returns a list of (destination, packet) transmissions.
        """
        transmissions = []
        for neighbor, queue in self.outgoing_queues.items():
            num_to_send = min(tx_rate_indirect, len(queue))
            sent_this_link = 0  # count how many packets we actually send to this neighbor
            for _ in range(num_to_send):
                packet = queue.pop(0)
                transmissions.append((neighbor, packet))
                self.active_buffer = [p for p in self.active_buffer if p.id != packet.id]
                self.tx_count += 1
                self.surface.get_node_by_id(neighbor).receive_packet(packet)
                sent_this_link += 1

            # If we sent anything on (self.id -> neighbor), record it in link_usage
            if sent_this_link > 0:
                self.surface.link_usage[(self.id, neighbor)] = (
                    self.surface.link_usage.get((self.id, neighbor), 0) + sent_this_link
                )
        return transmissions

    def update_load(self):
        """
        Updates local information and chooses the pertinent communication policy:
        0) Learning policy
        1) Line of Sight to lander, send directly there
        2) Spray and wait policy
        3) Greedy policy
        """
        # 0) Learning policy
        if self.surface.comm_policy == 2:
            self.learning_step()
        # 1) If direct lander contact, send directly (no copying needed).
        elif 0 in self.local_neighbors:
            if self.surface.line_of_sight_clear(self.position, self.surface.lander):
                self.send_packets_direct()
                self.outgoing_queues = {} # Destroy the queues, just sending to the lander
        # 2) Switch on Spray and Wait policy:
        elif self.surface.comm_policy == 1:
            self.spray_and_wait_step()
        # 3) Switch to Greedy policy. If indirect connection to the lander, find neighbors, otherwise nothing
        elif 0 in self.G_local:
            self.update_queues()
            self.process_queues()
        # if no connection to lander, skip
        else:
            pass

    def send_packets_direct(self):
        """
        If direct line-of-sight to the lander exists, send up to tx_rate_direct packets.
        For Spray and Wait it checks first the inactive buffer, since the oldest packets are there.
        """
        num_to_send = tx_rate_direct
        sent_to_lander = 0  # track how many we send this step

        # 1) Inactive buffer first
        while num_to_send > 0 and len(self.inactive_buffer) > 0:
            packet = self.inactive_buffer.pop(0)
            self.surface.lander_obj.receive_packet(packet, self.surface.sim_time)
            self.tx_count += 1
            num_to_send -= 1
            sent_to_lander += 1

        # 2) If capacity remains, send from active buffer
        while num_to_send > 0 and self.get_buffer_use() > 0:
            packet = self.active_buffer.pop(0)
            self.surface.lander_obj.receive_packet(packet, self.surface.sim_time)
            self.tx_count += 1
            num_to_send -= 1
            sent_to_lander += 1

        # If anything was sent, record it in link_usage for (self.id -> 0)
        if sent_to_lander > 0:
            self.surface.link_usage[(self.id, 0)] = (
                self.surface.link_usage.get((self.id, 0), 0) + sent_to_lander
            )

    def receive_packet(self, packet):
        """
        Add a received packet to the buffer. Under learning policy, if buffer full,
        drop the _new_ packet (not the oldest), record drop experience, and do NOT append.
        """
        packet.path.append([self.id, self.surface.sim_time])
        if self.is_buffer_full():
            self.drop_first()
            self.blocked_packet = True
        else:
            self.blocked_packet = False
        self.active_buffer.append(packet)
        # else:
        #     self.active_buffer.append(packet)
    
    def get_buffer_use(self):
        return len(self.active_buffer+self.inactive_buffer)
    
    def update_normalized_buffer_use(self):
        """
        Updates the rover's buffer usage normalized between 0 and max_buffer.
        0 = empty buffer, max_buffer = full buffer.
        """
        usage_ratio = self.get_buffer_use() / buffer_size  # buffer_size is a global param
        normalized_usage = usage_ratio * max_buffer
        self.buffer_normalized = normalized_usage
    
    # For Spray and Wait
    def is_buffer_full(self):
        """
        True if combined size of active+inactive buffers is >= buffer_size.
        """
        return (self.get_buffer_use()) >= buffer_size
    
    # Spray and Wait
    def spray_and_wait_step(self):
        """
        For each neighbor, send up to tx_rate_indirect copies of packets.
        If packet.copy_counter == 1, move it to inactive_buffer (no more replication).
        Skip neighbors that are full.
        Skip sending duplicates of a packet a neighbor already holds.
        Update link usage if we actually send a copy.
        """
        neighbor_ids = self.local_neighbors[:]
        random.shuffle(neighbor_ids)
        for neighbor_id in neighbor_ids:
        # for neighbor_id in self.local_neighbors:
            neighbor = self.surface.get_node_by_id(neighbor_id)
            neighbor = self.surface.get_node_by_id(neighbor_id)
            packets_sent = 0
            i = 0

            # We'll iterate through self.active_buffer from front to back
            while i < len(self.active_buffer) and packets_sent < tx_rate_indirect:
                packet = self.active_buffer[i]

                # If copy_counter=1, move packet to inactive buffer. No more replication.
                if packet.copy_counter == 1:
                    self.inactive_buffer.append(self.active_buffer.pop(i))
                    continue

                # If neighbor is full, skip this neighbor for this packet
                if neighbor.is_buffer_full():
                    i += 1
                    continue

                # Check for duplicates in neighbor's buffers
                if any(p.id == packet.id for p in neighbor.active_buffer) or \
                any(p.id == packet.id for p in neighbor.inactive_buffer):
                    # skip sending duplicates
                    i += 1
                    continue

                # Perform the "binary split" of copy_counter
                half = packet.copy_counter // 2
                packet.copy_counter -= half

                # Create new copy for neighbor
                new_packet = copy.deepcopy(packet)
                new_packet.copy_counter = half

                # Send the new copy
                neighbor.receive_packet(new_packet)
                self.tx_count += 1

                # Record link usage for plotting
                self.surface.link_usage[(self.id, neighbor_id)] = \
                    self.surface.link_usage.get((self.id, neighbor_id), 0) + 1

                # Advance counters
                i += 1
                packets_sent += 1
                self.copies += 1

# =============================================================================
# LunarSurface Class
# =============================================================================
class LunarSurface:
    def __init__(self, grd_size, paint_radius, comm_policy):
        """
        Initialize the lunar surface with grid resolution and rock parameters.
        Generates obstacles, sets the lander, creates a Lander object, instantiates rovers,
        and initializes metrics for delivered, dropped packets, and total flow.
        Also initializes independent RNGs for rocks and objectives.
        """
        self.sim_time = 0
        self.delivered_packets_over_time = []
        self.dropped_packets_over_time = []
        self.total_flow_over_time = []
        self.times = []
        self.comm_policy = comm_policy
        self.cumulative_flow = 0
        self.grd_size = grd_size
        self.paint_radius = paint_radius
        self.objective_rng = np.random.default_rng(objective_seed)
        self.lander = np.array(meters_to_normalized(lander_m[0], lander_m[1], map_origin, grd_size, resolution))
        self.generate_map(map_type)
        self.compute_lander_coverage_circle()
        self.clear_rocks_around_lander(radius=5)
        self.create_rovers()
        self.generate_graph()
        self.build_feasible_graph()
        self.link_usage = {}    # Dictionary mapping (senderID, receiverID) -> number_of_packets_sent
        self.movement_rng = np.random.default_rng(movement_seed)
        self.lander_obj = Lander(self.lander, self)
        self.finished = False

        if import_trajectories:
            print('-----------------------------------')
            print(f'Importing trajectories and objectives from {trajectories_local_path}')
            self.import_rover_trajectories_and_goals(trajectories_local_path)
            self.export_rover_trajectories(f'{outputPath}{local_trajectories_folder}')
            self.export_rover_goals(f'{outputPath}{local_trajectories_folder}')
        
        if comm_policy==2:
            self.DDQNAgent = RoverDDQNAgent(self)
            self.lander_obj.current_state, self.lander_obj.sorted_neighbors = self.DDQNAgent.observe_state(self.lander_obj) # Current state of the lander
            
            if import_models:
                self.load_models(models_local_path)
                print('-----------------------------------')
                print(f'Neural Networks imported from {models_local_path}!')
            else:
                self.DDQNAgent.q_network = self.DDQNAgent.build_model()
                self.DDQNAgent.q_target = self.DDQNAgent.build_model()
                print('-----------------------------------')
                print(f'Neural networks created!')
        else:
            self.DDQNAgent = None

    def check_finished_rovers(self):
        all_finished = True
        for rover in self.rovers:
            all_finished = all_finished and rover.finished
        if all_finished:
            print('-----------------------------------')
            print(f'All rovers finished exploring at step {self.sim_time}.')
            self.finished = True
        return True

    def compute_network_load_rho(self):
        """
        Computes average network load (rho) from simulation logs.
        Returns:
            rho_unique: Load based on unique packet generation
            rho_spray:  Load including spray copies (only if comm_policy==1)
        """
        lambda_unique = num_rovers / packet_generation_flow  # unique packets per time unit

        # Average number of rovers connected to the lander
        connected_time = 0
        for rover in self.rovers:
            rover_connected_time = sum(1 for status, _ in rover.connecteds if status == connected)
            connected_time += rover_connected_time

        A_avg = connected_time / sim_duration
        mu_total = A_avg * tx_rate_direct

        # Load using only unique packets
        rho_unique = lambda_unique / mu_total if mu_total > 0 else float('inf')

        if comm_policy == 1:  # Spray and Wait
            total_copies = sum(r.copies for r in self.rovers)
            total_originals = num_rovers * (sim_duration / packet_generation_flow)
            spray_factor = 1 + (total_copies / total_originals) if total_originals > 0 else 1
            lambda_spray = lambda_unique * spray_factor
            rho_spray = lambda_spray / mu_total if mu_total > 0 else float('inf')
            return rho_unique, rho_spray

        else:
            # For other policies, rho = rho_unique
            return rho_unique, rho_unique

    def save_models(self, outputPath):
        """
        Save the main DDQN models.
        """
        print('Saving Deep Neural networks at:', outputPath)
        os.makedirs(outputPath, exist_ok=True)

        # single, central agent that all rovers share
        agent = self.DDQNAgent
        n_rovers = len(self.rovers)

        agent.q_network.save(f"{outputPath}/qNetwork_{n_rovers}RVs.h5")
        if agent.ddqn:
            agent.q_target.save(f"{outputPath}/qTarget_{n_rovers}RVs.h5")

    def load_models(self, inputPath='./Models/'):
        """
        Load the main DDQN models.
        """
        print('Loading Deep Neural networks from:', inputPath)

        agent = self.DDQNAgent

        agent.q_network = load_model(f"{inputPath}/qNetwork_{num_rovers}RVs.h5")
        if agent.ddqn:
            agent.q_target = load_model(f"{inputPath}/qTarget_{num_rovers}RVs.h5")

    def get_node_by_id(self, rover_id):
        """
        Return the rover object in self.rovers that has the given id, or lander if if==0
        """
        return self.lander_obj if rover_id == 0 else next((r for r in self.rovers if r.id == rover_id), None)
    
    def export_rover_trajectories(self, output_path):
        os.makedirs(output_path, exist_ok=True)

        num_rovers = len(self.rovers)
        T = max(len(rover.trajectory) for rover in self.rovers)  # longest trajectory
        traj_matrix = []

        for t in range(T):
            row = []
            for rover in self.rovers:
                if t < len(rover.trajectory):
                    x, y = rover.trajectory[t]
                else:
                    x, y = np.nan, np.nan
                row.extend([t, x, y])
            traj_matrix.append(row)

        header = []
        for i in range(num_rovers):
            header.extend([f"time_{i}", f"x_{i}", f"y_{i}"])

        np.savetxt(f"{output_path}/trajectories.csv", traj_matrix, delimiter=",", header=",".join(header), comments='')

    def export_rover_goals(self, output_path):
        os.makedirs(output_path, exist_ok=True)

        num_rovers = len(self.rovers)
        max_len = max(len(rover.timed_goals) for rover in self.rovers)

        goal_matrix = []

        for j in range(max_len):
            row = []
            for rover in self.rovers:
                if j < len(rover.timed_goals):
                    t, x, y = rover.timed_goals[j]
                else:
                    t, x, y = np.nan, np.nan, np.nan
                row.extend([t, x, y])
            goal_matrix.append(row)

        header = []
        for i in range(num_rovers):
            header.extend([f"time_{i}", f"x_{i}", f"y_{i}"])

        np.savetxt(f"{output_path}/goals.csv", goal_matrix, delimiter=",", header=",".join(header), comments='')

    def import_rover_trajectories_and_goals(self, input_path):
        """
        Reads 'trajectories.csv' and 'goals.csv' from the given path,
        and sets rover.trajectory and rover.timed_goals for each rover.
        """
        traj_file = os.path.join(input_path, "trajectories.csv")
        goals_file = os.path.join(input_path, "goals.csv")

        traj_data = np.loadtxt(traj_file, delimiter=",", skiprows=1)
        goals_data = np.loadtxt(goals_file, delimiter=",", skiprows=1)

        num_rovers = len(self.rovers)

        # Restore trajectories
        for i, rover in enumerate(self.rovers):
            rover.trajectory = []
            time_idx = 3 * i     # time_i
            x_idx = 3 * i + 1    # x_i
            y_idx = 3 * i + 2    # y_i

            for row in traj_data:
                x, y = row[x_idx], row[y_idx]
                if not np.isnan(x) and not np.isnan(y):
                    rover.trajectory.append((x, y))

        # Restore timed goals
        for i, rover in enumerate(self.rovers):
            rover.timed_goals = []
            time_idx = 3 * i     # time_i
            x_idx = 3 * i + 1    # x_i
            y_idx = 3 * i + 2    # y_i

            for row in goals_data:
                t, x, y = row[time_idx], row[x_idx], row[y_idx]
                if not np.isnan(t) and not np.isnan(x) and not np.isnan(y):
                    rover.timed_goals.append((int(t), x, y))


            rover.timed_goals_index = 0

    def mark_rover_positions_congested(self):
        """
        Mark the area around each rover's position as congested (red = 2) if:
        - The cell is unexplored (ground != 1)
        - The cell is not a rock
        """
        for rover in self.rovers:
            r0, c0 = self.pos_to_cell(rover.position)
            for dr in range(-self.paint_radius, self.paint_radius + 1):
                for dc in range(-self.paint_radius, self.paint_radius + 1):
                    r, c = r0 + dr, c0 + dc
                    if 0 <= r < self.grd_size and 0 <= c < self.grd_size:
                        if self.rocks[r, c] == 0 and self.ground[r, c] != 1:
                            self.ground[r, c] = 2  # red = congested

    def process_packet_exploration_id_based_v0(self):
        """
        Process and remove newly received packets at the lander:
        - For each packet in `self.lander_obj.buffer`, check its ID.
        - If the packet ID is already in processed_buffer or duplicatedBuffer, move to duplicatedBuffer.
        - If the packet ID is new, mark it as explored and move to processed_buffer.
        - The packet is removed from the buffer regardless.
        """
        processed_ids = {pkt.id for pkt in self.lander_obj.processed_buffer}
        duplicated_ids = {pkt.id for pkt in self.lander_obj.duplicatedBuffer}

        for pkt in self.lander_obj.buffer:
            if pkt.id in processed_ids or pkt.id in duplicated_ids:
                self.lander_obj.duplicatedBuffer.append(pkt)
            else:
                self.lander_obj.processed_buffer.append(pkt)

        self.lander_obj.buffer.clear()

    def process_packet_exploration_id_based(self):
        """
        Process and remove newly received packets at the lander:
        - Use packet ID to detect duplicates.
        - If the packet is new, mark area as explored (paint green).
        - If already received, move to duplicatedBuffer.
        - Remove all packets from lander buffer.
        """
        processed_ids = {pkt.id for pkt in self.lander_obj.processed_buffer}
        duplicated_ids = {pkt.id for pkt in self.lander_obj.duplicatedBuffer}

        for pkt in self.lander_obj.buffer:
            if pkt.id in processed_ids or pkt.id in duplicated_ids:
                self.lander_obj.duplicatedBuffer.append(pkt)
            else:
                # Mark area around origin as explored (green)
                r0, c0 = self.pos_to_cell(pkt.origin)
                for dr in range(-self.paint_radius, self.paint_radius + 1):
                    for dc in range(-self.paint_radius, self.paint_radius + 1):
                        r, c = r0 + dr, c0 + dc
                        if 0 <= r < self.grd_size and 0 <= c < self.grd_size:
                            if self.rocks[r, c] == 0:
                                self.ground[r, c] = 1
                self.lander_obj.processed_buffer.append(pkt)

        self.lander_obj.buffer.clear()

    def process_packet_exploration_coordinate_based(self):
        """
        Process and remove newly received packets at the lander:
        - For each packet in `self.lander_obj.buffer`, check its origin.
        - If the area is already explored (ground == 1), move to duplicatedBuffer.
        - If unexplored, mark it as explored and move to processed_buffer.
        - The packet is removed from the buffer regardless.
        """
        for pkt in self.lander_obj.buffer:
            r0, c0 = self.pos_to_cell(pkt.origin)
            is_duplicate = True

            for dr in range(-self.paint_radius, self.paint_radius + 1):
                for dc in range(-self.paint_radius, self.paint_radius + 1):
                    r, c = r0 + dr, c0 + dc
                    if 0 <= r < self.grd_size and 0 <= c < self.grd_size:
                        if self.rocks[r, c] == 0 and self.ground[r, c] != 1:
                            self.ground[r, c] = 1
                            is_duplicate = False

            if is_duplicate:
                self.lander_obj.duplicatedBuffer.append(pkt)
            else:
                self.lander_obj.processed_buffer.append(pkt)

        self.lander_obj.buffer.clear()

    def create_rovers(self):
        # initialize L
        if comm_policy == 1:
            L = self.get_L_optimal(num_rovers)
        else:
            L = None
        self.rovers = []
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i in range(1, num_rovers+1):
            rover = Rover(i, self.lander, self, s_color=default_colors[(i-1) % len(default_colors)], L=L)
            # rover = Rover(i, self.lander, self)
            rover.generate_objective()
            rover.plan_path()
            self.rovers.append(rover)
    
    def generate_graph(self):
        self.G = nx.Graph()
        self.G.add_node(0)  # Lander node.
        for rover in self.rovers:
            self.G.add_node(rover.id)

    def clear_rocks_around_lander(self, radius=5):
        """
        Remove rocks within `radius` cells around the lander's grid cell.
        """
        # Convert lander's normalized position to grid indices
        r_lander, c_lander = self.pos_to_cell(self.lander)
        
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = r_lander + dr
                cc = c_lander + dc
                # Ensure (rr, cc) is within the map bounds
                if 0 <= rr < self.grd_size and 0 <= cc < self.grd_size:
                    # Clear rock
                    self.rocks[rr, cc] = 0  # set to 0 => no obstacle

    def generate_map(self, map_type):
        """
        Generates a map with obstacles. Two types of map can be generated:
        0: Map with random rocks. Less realistic.
        1: Realistic map with Lunar distribution generated. More realistic.
        """
        if map_type==0:
            self.rock_rng = np.random.default_rng(rock_seed)
        self.ground = np.zeros((grd_size, grd_size))
        if map_type == 0:
            self.rocks = np.zeros((grd_size, grd_size), dtype=int)
            self.generate_obstacles(num_rock_clusters, rock_cluster_radius)
        elif map_type == 1:
            _, grid_flat = get_map(resolution=resolution, map_width=grd_size, seed=map_seed, map_height=grd_size, ground_clearance_m=ground_clearance_m, max_rock_size=max_rock_size, cfa=cfa, crater_diameters=crater_diameters, crater_densities=crater_densities, output_path=outputPath)
            # self.ground = grid_flat.reshape((grd_size, grd_size))
            self.ground = np.flipud(grid_flat.reshape((grd_size, grd_size)))    # map is upside down
            self.rocks = (self.ground >= 100).astype(int)

    def generate_obstacles(self, num_rock_clusters, rock_cluster_radius):
        """
        Generate rock clusters on the surface.
        Each cluster is defined by a random center; all cells within the radius become obstacles.
        """
        for _ in range(num_rock_clusters):
            center_row = self.rock_rng.integers(0, self.grd_size)
            center_col = self.rock_rng.integers(0, self.grd_size)
            for i in range(self.grd_size):
                for j in range(self.grd_size):
                    if np.sqrt((i - center_row)**2 + (j - center_col)**2) <= rock_cluster_radius:
                        self.rocks[i, j] = 1

    def pos_to_cell(self, pos):
        """
        Convert a continuous position (in [0,1]^2) to grid cell indices.
        """
        col = int(pos[0] * self.grd_size)
        row = int(pos[1] * self.grd_size)
        return (min(row, self.grd_size-1), min(col, self.grd_size-1))

    def line_of_sight_clear(self, pos1, pos2, num_samples=50):
        """
        Check if the straight line between pos1 and pos2 is free of obstacles.
        Samples points along the line; returns False if any point lies on a rock.
        """
        for t in np.linspace(0, 1, num_samples):
            p = pos1 + t * (pos2 - pos1)
            r, c = self.pos_to_cell(p)
            if self.rocks[r, c] == 1:
                return False
        return True

    def update_ground(self):
        """
        Update the explored ground by marking grid cells that rovers pass over.
        - Green (1) if rover is not congested.
        - Red   (3) if rover is fully congested.
        """
        for rover in self.rovers:
            r0, c0 = self.pos_to_cell(rover.position)
            for dr in range(-self.paint_radius, self.paint_radius + 1):
                for dc in range(-self.paint_radius, self.paint_radius + 1):
                    r = r0 + dr
                    c = c0 + dc
                    if 0 <= r < self.grd_size and 0 <= c < self.grd_size:
                        if self.rocks[r, c] == 0:
                            if rover.is_buffer_full() and self.ground[r, c] != 1: # congested if full and unexplored
                                self.ground[r, c] = 2  # red = congested
                            else:
                                self.ground[r, c] = 1  # green = normal explored

    def update_rovers(self, step_size):
        """
        Update the position of each rover.
        """
        for rover in self.rovers:
            rover.update_position(step_size)

    def update_graph_edges(self, threshold_rover, threshold_base):
        """
        Rebuild the communication graph based on Euclidean distance.
        Nodes include the lander and all rovers.
        """
        self.G.remove_edges_from(list(self.G.edges()))
        nodes_positions = {0: self.lander}
        for rover in self.rovers:
            nodes_positions[rover.id] = rover.position
        nodes = list(nodes_positions.keys())
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                th = threshold_base if (nodes[i] == 0 or nodes[j] == 0) else threshold_rover
                if np.linalg.norm(nodes_positions[nodes[i]] - nodes_positions[nodes[j]]) < th:
                    self.G.add_edge(nodes[i], nodes[j])
        return self.G

    def build_edge_lists(self, num_samples=50):
        """
        Build two lists of edges:
          - Feasible edges: those with clear line-of-sight.
          - Obstructed edges: those with an obstacle between.
        """
        feasible_edges = []
        obstructed_edges = []
        nodes_positions = {0: self.lander}
        for rover in self.rovers:
            nodes_positions[rover.id] = rover.position
        for edge in self.G.edges():
            u, v = edge
            if self.line_of_sight_clear(nodes_positions[u], nodes_positions[v], num_samples):
                feasible_edges.append(edge)
            else:
                obstructed_edges.append(edge)
        return feasible_edges, obstructed_edges

    def build_feasible_graph(self):
        """
        Builds the feasible graph. Basically remove the edges that are blocked by rocks.
        """
        feasible_edges, _ = self.build_edge_lists()
        G_feasible = nx.Graph()
        G_feasible.add_nodes_from(self.G.nodes())
        G_feasible.add_edges_from(feasible_edges)
        self.G_feasible = G_feasible

    def update_loads(self):
        """
        Process communications according to the selected communication policy.
        If the learning policy is selected, update the observed state by nodes.
        """
        if comm_policy == 2:
            self.lander_obj.update_local_graph_lander()                                     # Update surface status
            self.lander_obj.current_state, self.lander_obj.sorted_neighbors = self.DDQNAgent.observe_state(self.lander_obj)   # Update surface state
            for rover in self.rovers:        
                rover.last_state = rover.current_state                                      # Update rovers status
                rover.compute_ttl()
                rover.update_normalized_buffer_use()
                if HTL:
                    rover.update_hops_to_lander()
        for rover in self.rovers:        
            rover.update_connected()

        if comm_policy == 2:
            for rover in self.rovers:                                                       # Update rovers state
                rover.current_state, rover.sorted_neighbors = self.DDQNAgent.observe_state(rover)

        for rover in self.rovers:
            rover.update_load()

    def update_rover_neighbors(self):
        """
        Update the local subgraph and direct neighbors for each rover.
        This is done after the feasible graph is built.
        """
        for rover in self.rovers:
            rover.update_local_subgraph()                                                   # Update rovers status
            rover.update_direct_neighbors()    

    def update(self, step_size):
        """
        Perform a full update of the lunar surface:
          - Update rover positions.
          - Builds graph and feasible graph (excluding links obstructed by obstacles like rocks)
          - Update the communication graph and process communications.
          - Update the explored ground.
          - Record metrics: delivered, dropped (effective), and total flow. NO
          - Increment simulation time by time_interval.
        """
        self.link_usage = {}
        self.update_rovers(step_size)
        self.update_graph_edges(threshold_rover, threshold_base)
        self.build_feasible_graph()
        self.update_rover_neighbors()
        self.update_loads()
        if packet_coordinates:
            self.process_packet_exploration_id_based()
            self.mark_rover_positions_congested()
        else:
            self.update_ground()
        self.sim_time += time_interval
    
    def compute_lander_coverage_circle(self, radius=threshold_base, num_points=500):
        angles = np.linspace(0, 2 * np.pi, num_points)
        segments = []
        current_segment = []
        current_type = None  # 'los' or 'blocked'

        for theta in angles:
            x = self.lander[0] + radius * np.cos(theta)
            y = self.lander[1] + radius * np.sin(theta)
            if not (0 <= x <= 1 and 0 <= y <= 1):
                continue

            target = np.array([x, y])
            is_los = self.line_of_sight_clear(self.lander, target)
            seg_type = 'los' if is_los else 'blocked'

            if current_type is None:
                current_type = seg_type
                current_segment = [target]
            elif seg_type == current_type:
                current_segment.append(target)
            else:
                # Save current segment and start a new one
                if current_segment:
                    segments.append((current_type, current_segment))
                current_type = seg_type
                current_segment = [target]

        # Save last segment
        if current_segment:
            segments.append((current_type, current_segment))

        self.coverage_segments = segments
    
    def prepare_display(self, ax, combine_buffers=True, dropped_counter=False):
        """
        Prepare the display:
        - Draw the ground (explored areas and rocks).
        - Draw communication edges (feasible in gray, obstructed in red).
        - Draw nodes (lander and rovers) with colors based on local buffer length
            but an edge outline with rover.static_color (thicker outline).
        - Mark rover objectives.
        - If combine_buffers=True, display only the total buffer in blue.
            Otherwise, display two numbers: the total buffer in blue (above),
            and the inactive buffer in black (just below).
        - If dropped_counter = False, Additionally, show a third number (top-left of the rover) indicating
            how many packets the rover has dropped so far.
        - Highlight used edges in blue and label directional usage counts
            with the static color of the source rover.
        - Add legend and timestamp.
        """

        if comm_policy == 0:        # Greedy policy does not have such an inactive buffer
            combine_buffers = True

        # Convert map arrays for plotting
        display_grid = self.ground.copy()
        display_grid[self.rocks == 1] = 3  # "2" = rock
        ground_cmap = ListedColormap(['white', 'green', 'red', 'black']) # #FF0000 -> strong red
        self.im = ax.imshow(display_grid, extent=[0, 1, 0, 1], origin='lower', cmap=ground_cmap, alpha=1)

        # Build feasible/obstructed edges
        feasible_edges, obstructed_edges = self.build_edge_lists()
        nodes_positions = {0: self.lander}
        for rover in self.rovers:
            nodes_positions[rover.id] = rover.position

        # Draw feasible (gray) and obstructed (red) edges
        nx.draw_networkx_edges(self.G, nodes_positions, edgelist=feasible_edges, ax=ax, edge_color='gray')
        nx.draw_networkx_edges(self.G, nodes_positions, edgelist=obstructed_edges, ax=ax, edge_color='red')

        # Prepare node colors
        node_colors = {}
        static_colors = {}
        # Lander color
        node_colors[0] = "#D8BFD8"
        static_colors[0] = "#D8BFD8"
        # Rover colors
        for rover in self.rovers:
            node_colors[rover.id] = rover.color           # color based on buffer length
            static_colors[rover.id] = rover.static_color  # thicker outline color

        # Draw nodes with thicker outline
        nx.draw_networkx_nodes(
            self.G,
            nodes_positions,
            nodelist=list(self.G.nodes()),
            node_color=[node_colors[n] for n in self.G.nodes()],
            edgecolors=[static_colors[n] for n in self.G.nodes()],
            linewidths=2,  # Thicker outline
            ax=ax,
            node_size=300
        )

        # Label only the lander
        nx.draw_networkx_labels(self.G, nodes_positions, {0: "Lander"}, ax=ax)

        # Mark objectives and display rover buffer usage
        for rover in self.rovers:
            ax.plot(rover.objective[0], rover.objective[1], marker='x', color='magenta', markersize=10)
            offset = 0.01

            if combine_buffers:
                # Just the total buffer usage in blue
                ax.text(
                    rover.position[0],
                    rover.position[1],
                    f"{rover.get_buffer_use()}",
                    fontsize=8,
                    color='blue'
                )
            else:
                # Two numbers, top = total (blue), bottom = inactive (black)
                # top line: total buffer usage
                ax.text(
                    rover.position[0],
                    rover.position[1] + offset,
                    f"{rover.get_buffer_use()}",
                    fontsize=8,
                    color='blue',
                    ha='center'
                )
                # bottom line: inactive buffer usage
                if comm_policy == 1:
                    ax.text(
                        rover.position[0],
                        rover.position[1] - offset,
                        f"{len(rover.inactive_buffer)}",
                        fontsize=8,
                        color='black',
                        ha='center'
                    )
            # --- show rover ID at center of the rover if comm policy is not Spray and Wait ---
            if comm_policy != 1:
                ax.text(
                    rover.position[0],
                    rover.position[1] - offset,
                    f"{rover.id}",
                    fontsize=8,
                    color='black',
                    fontweight='bold',
                    ha='center'
                )

            # --- show dropped packets at top-left of the rover position ---
            if dropped_counter:
                dropped_count = sum(1 for p in rover.dropped_packets if p.dropped_time is not None)
                # if dropped_count > 0:
                offset_tl = 0.02  # shift left & up
                ax.text(
                    rover.position[0] - offset_tl,
                    rover.position[1] + offset_tl,
                    f"{dropped_count}",
                    fontsize=8,
                    color='red',
                    ha='right',   # anchor text's right edge at (x - offset_tl)
                    va='bottom'   # anchor text's bottom edge at (y + offset_tl)
                )

            # --- show Time To Lander (TTL) at top-right of the rover position ---
            if show_ttl and comm_policy == 2:
                ttl_value = rover.ttl
                offset_tr = 0.02  # shift right & up
                ax.text(
                    rover.position[0] + offset_tr,
                    rover.position[1] + offset_tr,
                    f"{ttl_value:.2F}",
                    fontsize=8,
                    color='purple',
                    ha='left',    # anchor text's left edge at (x + offset_tr)
                    va='bottom'   # anchor text's bottom edge at (y + offset_tr)
                )

            # --- show Hops To Lander (HTL) at bottom-right of the rover position ---
            if HTL and comm_policy == 2:
                htl_value = rover.htl
                offset_br = 0.02  # shift right & down
                ax.text(
                    rover.position[0] + offset_br,
                    rover.position[1] - offset_br,
                    f"{htl_value}",
                    fontsize=8,
                    color='purple',
                    ha='left',    # anchor text's left edge at (x + offset_br)
                    va='top'      # anchor text's top edge at (y - offset_br)
                )

            # --- indicate finished rovers ---
            if rover.finished:
                ax.plot(
                    rover.position[0],
                    rover.position[1],
                    marker='*',
                    color='green',
                    markersize=12,
                    markeredgewidth=1.5,
                    markeredgecolor='white'
                )
        
        # Draw segmented lander coverage circle
        if hasattr(self, 'coverage_segments'):
            for seg_type, segment in self.coverage_segments:
                color = 'gray' if seg_type == 'los' else 'red'
                ax.plot([p[0] for p in segment], [p[1] for p in segment], color=color, linewidth=1.5)

        # Add legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='k', label='Lander', markerfacecolor="#D8BFD8", markersize=10),
            Line2D([0], [0], marker='o', color='k', label='Rover', markerfacecolor="#ADD8E6", markersize=10),
            Patch(facecolor='green', edgecolor='k', label='Explored Area'),
            Patch(facecolor='red', edgecolor='k', label='Visited Area'),
            Patch(facecolor='black', edgecolor='k', label='Rock'),
            Line2D([0], [0], marker='x', color='magenta', label='Objective', markersize=10, linestyle='None'),
            Line2D([0], [0], color='gray', marker='o', linestyle='None', markerfacecolor='none', markersize=10, label='Lander Coverage')
        ]
        ax.legend(handles=legend_elements, loc='lower left')

        # Timestamp in the corner
        if comm_policy==2:
            time_text = f"Time: {self.sim_time}s\nEpsilon: {self.DDQNAgent.epsilon:.2f}"
        else:
            time_text = f"Time: {self.sim_time}s"
        ax.text(0.03, 0.95, time_text, transform=ax.transAxes, fontsize=12, verticalalignment='top')

        # Highlight edges used this timestep in blue, then label directional usage counts
        # with the static color of the source rover.
        used_edges = set()
        for (u, v), packets_sent_uv in self.link_usage.items():
            if packets_sent_uv > 0:
                used_edges.add((u, v))

        # Draw a "blue" edge if there's usage in at least one direction
        edges_to_draw_in_blue = set()
        for (u, v) in used_edges:
            edges_to_draw_in_blue.add((u, v))
            if (v, u) in used_edges:
                edges_to_draw_in_blue.add((v, u))

        nx.draw_networkx_edges(
            self.G,
            nodes_positions,
            edgelist=list(edges_to_draw_in_blue),
            ax=ax,
            edge_color='blue',
            width=2
        )

        # Add directional usage text
        for (u, v), packets_sent_uv in self.link_usage.items():
            if packets_sent_uv == 0:
                continue
            packets_sent_vu = self.link_usage.get((v, u), 0)

            x_u, y_u = nodes_positions[u]
            x_v, y_v = nodes_positions[v]
            x_mid = 0.5 * (x_u + x_v)
            y_mid = 0.5 * (y_u + y_v)

            dx = x_v - x_u
            dy = y_v - y_u
            length = max(1e-6, (dx**2 + dy**2) ** 0.5)
            px = dy / length
            py = -dx / length
            offset_factor = 0.02

            # Usage text for u->v
            x_label_pos_uv = x_mid + offset_factor * px
            y_label_pos_uv = y_mid + offset_factor * py
            source_color_uv = static_colors.get(u, '#000000')

            ax.text(
                x_label_pos_uv,
                y_label_pos_uv,
                f"{packets_sent_uv}",
                color=source_color_uv,
                fontsize=8,
                fontweight='bold',
                ha="center"
            )

            # If usage in other direction, label that
            if packets_sent_vu > 0:
                x_label_pos_vu = x_mid - offset_factor * px
                y_label_pos_vu = y_mid - offset_factor * py
                source_color_vu = static_colors.get(v, '#000000')

                ax.text(
                    x_label_pos_vu,
                    y_label_pos_vu,
                    f"{packets_sent_vu}",
                    color=source_color_vu,
                    fontsize=8,
                    fontweight='bold',
                    ha="center"
                )

    def get_L_optimal(self, num_nodes: int) -> int:
        """
        Calculate the optimal number of copies (L) for the spray-and-wait protocol
        based on the total number of nodes in the network.
        
        The heuristic used here is: L = ceil(sqrt(num_nodes))
        """
        L = math.ceil(math.sqrt(num_nodes))
        print(f'L: {L}')
        return L

    def plot_save_results(self, plot_per_rover=False, avg_window=30, folder_figs='figures', folder_csv = f'csv_{comm_policy_str}'):
        """
        Create three plots:
        1. Dropped Packets CDF
        2. Dropped packets CDF and dropped packets per node
        3. Average Packet Delay Over Time
        """
        print('Saving results figures in:', folder_figs)
        
        # --- folders ---
        outputPathFigures = os.path.join(outputPath, folder_figs)   # “…/results/…/figures”
        outputPathcsv    = os.path.join(outputPath, folder_csv)     # “…/results/…/csv_*”
        os.makedirs(outputPathFigures, exist_ok=True)
        os.makedirs(outputPathcsv,  exist_ok=True)

        # Configurable average window (in number of time intervals)
        avg_window = 30
        # Collect drop times from all rovers and per rover.
        all_drop_times = []
        per_rover_drop_times = {}
        for rover in self.rovers:
            drop_times = [pkt.dropped_time for pkt in rover.dropped_packets if pkt.dropped_time is not None]
            drop_times.sort()
            all_drop_times.extend(drop_times)
            per_rover_drop_times[rover.id] = drop_times

        all_drop_times.sort()
        if not all_drop_times:
            print("No dropped packets to plot.")
            return
        
        # Define time bins
        bins = np.arange(0, self.sim_time + time_interval, time_interval)
        counts_all, _ = np.histogram(all_drop_times, bins=bins)
        # ----------------------------
        # Plot 1: CDF of Dropped Packets
        # ----------------------------
        cumulative_all = np.cumsum(counts_all)
        total_dropped = cumulative_all[-1]
        cdf_all = cumulative_all / total_dropped if total_dropped > 0 else cumulative_all

        # Create a color mapping for each rover
        if plot_per_rover:
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            rover_colors = {rid: color_cycle[i % len(color_cycle)] for i, rid in enumerate(per_rover_drop_times.keys())}

        plt.figure(dpi=500)
        plt.plot(bins[:-1], cdf_all, label="Total", color='blue')
        if plot_per_rover:
            for rid, drop_times in per_rover_drop_times.items():
                counts, _ = np.histogram(drop_times, bins=bins)
                cumulative = np.cumsum(counts)
                cdf = cumulative / total_dropped if total_dropped > 0 else cumulative
                plt.plot(bins[:-1], cdf, linestyle='--', label=f"Rover {rid}", color=rover_colors[rid], linewidth=0.8, alpha=0.5)
        plt.xlabel("Time (s)")
        plt.ylabel("CDF")
        plt.title("Dropped Packets CDF")
        plt.legend()
        plt.savefig(f"{outputPathFigures}/1_cdf_dropped_packets.png", dpi=500)
        plt.close()

        # ----------------------------
        # Plot 2: Dropped packets CDF and dropped packets per node
        # ----------------------------
        per_rover_window   = {}
        # window_bin_centers = []
        counts_all, _ = np.histogram(all_drop_times, bins=bins)
        cumulative_all = np.cumsum(counts_all)
        total_dropped = cumulative_all[-1]
        cdf_all = cumulative_all / total_dropped if total_dropped > 0 else cumulative_all


        fig, ax1 = plt.subplots(dpi=500)
        ax1.plot(bins[:-1], cdf_all, label="Total CDF", color='blue')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("CDF", color='black')
        ax1.tick_params(axis='y', labelcolor='black')

        ax2 = ax1.twinx()
        if plot_per_rover:
            per_rover_window = {}
            for rid, drop_times in per_rover_drop_times.items():
                counts, _ = np.histogram(drop_times, bins=bins)
                # ax2.plot(bins[:-1], counts, linestyle='--', label=f"Rover {rid}", color=rover_colors[rid], linewidth=0.8, alpha=0.5)
                window_counts = []
                window_bin_centers = []
                for i in range(len(bins) - avg_window):
                    window_counts.append(np.sum(counts[i:i+avg_window]))
                    window_bin_centers.append((bins[i] + bins[i + avg_window]) / 2)
                ax2.plot(window_bin_centers, window_counts, linestyle='--', label=f"Rover {rid}", 
                        color=rover_colors[rid], linewidth=0.8, alpha=0.5)
                per_rover_window[rid] = window_counts

        ax2.set_ylabel("Dropped Packets", color='black')
        ax2.tick_params(axis='y', labelcolor='black')

        # Combine legends from both axes.
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.title(f"Dropped packets per {avg_window}s")
        plt.savefig(f"{outputPathFigures}/2_dropped_packets_per_{avg_window}s.png", dpi=500)
        plt.close()

        # ----------------------------
        # Plot 3: Average Packet Delay Over Time (Single Y-Axis)
        # ----------------------------
        # Gather delivered packets (those with a recorded delivery time)
        delivered_packets = [pkt for pkt in self.lander_obj.processed_buffer if pkt.delivered_to_lander_time is not None]
        delivered_packets_duplicated = [pkt for pkt in self.lander_obj.duplicatedBuffer if pkt.delivered_to_lander_time is not None]
        sorted_delivered = sorted(delivered_packets + delivered_packets_duplicated, key=lambda pkt: pkt.delivered_to_lander_time)


        if not sorted_delivered:
            print("No delivered packets to plot for delay.")
        else:
            bins = np.arange(0, self.sim_time + time_interval, time_interval)
            agg_avg_delay = []
            bin_centers = []
            # Compute aggregate average delay using a sliding window
            for i in range(len(bins) - avg_window):
                window_packets = [pkt for pkt in delivered_packets if bins[i] <= pkt.creation_time < bins[i+avg_window]]
                if window_packets:
                    delays = [pkt.delivered_to_lander_time - pkt.creation_time for pkt in window_packets]
                    agg_avg_delay.append(np.mean(delays))
                else:
                    agg_avg_delay.append(0)
                bin_centers.append((bins[i] + bins[i+avg_window]) / 2)
            
            # Compute per-rover sliding window average delays
            per_rover_avg_delay = {}
            for rover in self.rovers:
                rover_packets = [pkt for pkt in delivered_packets if pkt.source_id == rover.id]
                rover_avg_delay = []
                for i in range(len(bins) - avg_window):
                    window_packets = [pkt for pkt in rover_packets if bins[i] <= pkt.creation_time < bins[i+avg_window]]
                    if window_packets:
                        delays = [pkt.delivered_to_lander_time - pkt.creation_time for pkt in window_packets]
                        rover_avg_delay.append(np.mean(delays))
                    else:
                        rover_avg_delay.append(0)
                per_rover_avg_delay[rover.id] = rover_avg_delay
            
            # Create a single-axis plot for average delay
            fig, ax = plt.subplots(dpi=500)
            ax.plot(bin_centers, agg_avg_delay, label="Total", color='blue')
            if plot_per_rover:
                color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
                for i, (rid, delays) in enumerate(per_rover_avg_delay.items()):
                    ax.plot(bin_centers, delays, linestyle='--', label=f"Rover {rid}", 
                            color=color_cycle[i % len(color_cycle)], linewidth=0.8, alpha=0.5)
            ax.set_xlabel("Packet Creation Time (s)")
            ax.set_ylabel("Average Delay (s)")
            ax.legend(loc='upper left')
            plt.title("Average Packet Delay Over Time")
            plt.savefig(f"{outputPathFigures}/3_average_delay.png", dpi=500)
            plt.close()

        # ----------------------------
        # Plot 4: Buffer Usage (Congestion Level) Over Time
        # ----------------------------
        fig, ax = plt.subplots(dpi=500)

        # Collect all buffer usage traces as percentage
        buffer_matrix = {}
        for rover in self.rovers:
            usages, steps = zip(*rover.buffer_use)
            usages_pct = [100 * u / buffer_size for u in usages]
            ax.plot(steps, usages_pct, label=f'Rover {rover.id}', linewidth=1, linestyle='--')
            for t, u_pct in zip(steps, usages_pct):
                buffer_matrix.setdefault(t, []).append(u_pct)

        # Compute average usage percentage per timestep
        avg_steps = sorted(buffer_matrix.keys())
        avg_usages_pct = [np.mean(buffer_matrix[t]) for t in avg_steps]
        ax.plot(avg_steps, avg_usages_pct, label='Average', linewidth=2, color='black')

        # Save CSV
        np.savetxt(f"{outputPathcsv}/average_buffer.csv",
                np.vstack([avg_steps, avg_usages_pct]).T,
                delimiter=",", header="time,avg_buffer_pct", comments='')

        ax.set_xlabel('Time Step (s)')
        ax.set_ylabel('Buffer Usage (%)', color='black')
        ax.tick_params(axis='y', labelcolor='black')
        ax.set_title('Buffer Usage Over Time per Rover')
        ax.legend(loc='upper right')
        plt.savefig(f"{outputPathFigures}/4_congestion_buffer_usage.png", dpi=500)
        plt.close()



        # ----------------------------
        # Plot 5: Effective Delivered, Dropped, and Duplicated Packets Over Time
        # ----------------------------
        max_time = int(self.sim_time) + 1
        delivered_over_time = np.zeros(max_time)
        dropped_over_time = np.zeros(max_time)
        duplicated_over_time = np.zeros(max_time)

        for pkt in self.lander_obj.processed_buffer:
            if pkt.delivered_to_lander_time is not None:
                t = int(pkt.delivered_to_lander_time)
                if t < max_time:
                    delivered_over_time[t] += 1

        for pkt in self.lander_obj.duplicatedBuffer:
            if pkt.delivered_to_lander_time is not None:
                t = int(pkt.delivered_to_lander_time)
                if t < max_time:
                    duplicated_over_time[t] += 1

        for rover in self.rovers:
            for pkt in rover.dropped_packets:
                if pkt.dropped_time is not None:
                    t = int(pkt.dropped_time)
                    if t < max_time:
                        dropped_over_time[t] += 1

        # Compute cumulative values
        delivered_cum = np.cumsum(delivered_over_time)
        dropped_cum = np.cumsum(dropped_over_time)
        duplicated_cum = np.cumsum(duplicated_over_time)

        # Total packets = delivered + dropped + duplicated
        total_cum = delivered_cum + dropped_cum + duplicated_cum

        # Plot
        plt.figure(dpi=500)
        plt.plot(np.arange(max_time), delivered_cum, label='Delivered (Unique)', color='green')
        plt.plot(np.arange(max_time), dropped_cum, label='Dropped', color='red')
        plt.plot(np.arange(max_time), duplicated_cum, label='Duplicated', color='orange')
        plt.plot(np.arange(max_time), total_cum, label='Total', color='blue', linestyle='--')

        plt.xlabel('Time (s)')
        plt.ylabel('Cumulative Packets')
        plt.title('Delivered, Dropped and Duplicated CDFs over time')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{outputPathFigures}/5_effective_packets_over_time.png", dpi=500)
        plt.close()

        print('Saving data results in:', outputPathcsv)

        # 2‑A  dropped times ----------------------------------------------------------
        np.savetxt(f"{outputPathcsv}/all_drop_times.csv", all_drop_times, delimiter=",")

        for rid, drop in per_rover_drop_times.items():
            np.savetxt(f"{outputPathcsv}/drop_times_r{rid}.csv", drop, delimiter=",")

        # 2‑B  CDF counts -------------------------------------------------------------
        np.savetxt(f"{outputPathcsv}/cdf_total.csv",
                np.vstack([bins[:-1], cdf_all]).T, delimiter=",",
                header="time,cdf", comments='')

        # 2‑C  sliding‑window drop counts --------------------------------------------
        if plot_per_rover: 
            for rid, window_counts in per_rover_window.items():
                np.savetxt(f"{outputPathcsv}/drops_window_r{rid}.csv",
                        np.vstack([window_bin_centers, window_counts]).T,
                        delimiter=",", header="time,drops", comments='')

        # 2‑D  average delay ----------------------------------------------------------
        if sorted_delivered:
            np.savetxt(f"{outputPathcsv}/avg_delay_total.csv",
                    np.vstack([bin_centers, agg_avg_delay]).T,
                    delimiter=",", header="time,avg_delay", comments='')

        # 2‑E  buffer use -------------------------------------------------------------
        for rover in self.rovers:
            use, step = zip(*rover.buffer_use)
            np.savetxt(f"{outputPathcsv}/buffer_r{rover.id}.csv",
                    np.vstack([step, use]).T,
                    delimiter=",", header="step,buffer_use", comments='')

        # 2‑F  delivered / dropped / duplicated CDFs ----------------------------------
        np.savetxt(f"{outputPathcsv}/effective_packets_cdf.csv",
                np.vstack([np.arange(max_time),
                            delivered_cum,
                            dropped_cum,
                            duplicated_cum,
                            total_cum]).T,
                delimiter=",",
                header="time,delivered,dropped,duplicated,total",
                comments='')

    def plot_buffer_usage_penalty(self, alpha_values, penalty_drop=-50, base=10, resolution=100, output_path="./"):
        """
        Plot buffer penalty vs. usage ratio for a list of alpha values and save the plot to a file.
        """
        print('Saving buffer usage reward function:', output_path + '/buffer_usage_penalty.png')
        os.makedirs(output_path, exist_ok=True)

        usage_ratios = np.linspace(0, 1, resolution)
        plt.figure(dpi=120)

        for alpha in alpha_values:
            penalties = penalty_drop * (base**(alpha * usage_ratios) - 1) / (base**alpha - 1)
            plt.plot(usage_ratios, penalties, label=f'Alpha = {alpha}', linewidth=2)

        plt.xlabel("Buffer Usage Ratio", fontsize=18)
        plt.ylabel("Penalty", fontsize=18)
        plt.title("Buffer Usage Penalty", fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.legend(fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path + '/buffer_usage_penalty.png', dpi=300)
        plt.close()

# =============================================================================
# Main Animation Function
# =============================================================================
def animate(frame, surface, show_display=True):
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    surface.update(step_size)
    if show_display:
        surface.prepare_display(ax, combine_buffers=False, dropped_counter=True)
    print_progress(frame + 1, frames_count)
    return []

def print_progress(current, total, bar_length=40):
    progress = current / total
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f'\rProgress |{bar}| {current}/{total} steps ({progress:.1%})', end='')
    if current == total:
        print()  # new line when complete


# =============================================================================
# Main Execution: Set up Animation or Run Simulation
# =============================================================================
if __name__ == "__main__":

    startTime = time.time()
    sys.stdout = Logger(outputPath + 'logfile.log')

    fig, ax = plt.subplots(figsize=(6,6))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)   # REVIEW title can not be seen with this
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Create the LunarSurface instance.
    print('-----------------------------------')
    print(f'Generating Lunar Surface with {num_rovers} rovers with buffer size: {buffer_size}')
    surface = LunarSurface(grd_size, paint_radius, comm_policy)

    if not import_trajectories and traj_type == 1:
        print('-----------------------------------')
        print(f'Generating trajectories and objectives with PSE planner')
        assign_docker_trajectories_and_goals(surface, outputPath)
        plot_final_trajectories(surface, outputPath)
    
    elif not import_trajectories and traj_type == 2:
        print('-----------------------------------')
        print(f'Generating trajectories following CADRE planner paper')
        run_exploration(surface, output_dir=outputPath)
        plot_final_trajectories(surface, outputPath)

    print('-----------------------------------')
    print(f'Simulation started!\nCommunications policy: {comm_policy_str}\nDuration: {sim_duration}\nMap type: {map_type_str}\nVideo option: {video_option}')
    print('-----------------------------------')

    if video_option == "display":
        ani = FuncAnimation(fig, animate, fargs=(surface,), frames=frames_count, interval=1, blit=False, init_func=lambda: [])
        plt.show()
    elif video_option == "save":
        ani = FuncAnimation(fig, animate, fargs=(surface,), frames=frames_count, interval=1, blit=False, init_func=lambda: [])
        writer = FFMpegWriter(fps=30, metadata=dict(artist='Federico Lozano-Cuadra'), bitrate=1800)
        ani.save(f"{outputPath}/rover_animation.mp4", writer=writer)
    elif video_option == "no video":
        # Run simulation without generating video.
        for frame in range(frames_count):
            surface.update(step_size)
            print_progress(frame + 1, frames_count)
            if surface.finished:
                break

    # compute rho
    rho_unique, rho_spray = surface.compute_network_load_rho()
    print(f"\nrho (unique packets): {rho_unique:.4f}, rho (including spray copies): {rho_spray:.4f}")


    # Store and plot trajectories and goals
    if import_trajectories or traj_type == 0:
        plot_final_trajectories(surface, outputPath)

    # Once done, print a newline
    print()

    surface.plot_save_results(plot_per_rover=True)
    if comm_policy == 2:
        surface.save_models(f'{outputPath}/Models/')
        surface.plot_buffer_usage_penalty(alpha_values=[buffer_steppness], penalty_drop=penalty_drop, output_path=f'{outputPath}/figures/')

    print(f'Simulation finished! Elapsed time: {(time.time() - startTime):.2F}s')

