#! /usr/bin/env python
"""
 Copyright 2023 by California Institute of Technology.  ALL RIGHTS RESERVED.
 United  States  Government  sponsorship  acknowledged.   Any commercial use
 must   be  negotiated  with  the  Office  of  Technology  Transfer  at  the
 California Institute of Technology.
 
 This software may be subject to  U.S. export control laws  and regulations.
 By accepting this document,  the user agrees to comply  with all applicable
 U.S. export laws and regulations.  User  has the responsibility  to  obtain
 export  licenses,  or  other  export  authority  as may be required  before
 exporting  such  information  to  foreign  countries or providing access to
 foreign persons.
 
 This  software  is a copy  and  may not be current.  The latest  version is
 maintained by and may be obtained from the Mobility  and  Robotics  Sytstem
 Section (347) at the Jet  Propulsion  Laboratory.   Suggestions and patches
 are welcome and should be sent to the software's maintainer.
 
"""

import random
import math
import argparse
import csv
import os
import subprocess
from map_generation.MapGenerator import MapGenerator
import numpy as np
# from map_src.write_csv_file import write_to_csv
import pandas as pd
from multiprocessing import Pool
import pathlib
import time
from tqdm import tqdm

PARALLEL = True
VERBOSE = False

def print_rover_locations_vnl(location_file_name, locations_list):
    if len(locations_list):
        with open(location_file_name, 'w', encoding="utf-8") as file:
            file.write("# t n")
            for follower_ix in range(len(locations_list[0])):
                file.write(" f{i}_x f{i}_y f{i}_theta f{i}_dr f{i}_dyaw".format(i=follower_ix))
            file.write('\n')
            for locations_ix, locations in enumerate(locations_list):
                file.write("{} {}".format(
                    locations_ix,
                    len(locations),
                ))
                for follower_ix in range(len(locations)):
                    file.write(" {} {} {} {} {}".format(
                            locations[follower_ix][0],
                            locations[follower_ix][1],
                            locations[follower_ix][2],
                            0.,
                            0.,
                        )
                    )
                file.write("\n")

def run_one_monte_carlo(
        executable: str="../build/bin/libcadre-formation-sensing-cli",
        seed: int=None,
        suffix: str="",
        resolution: float=0.05,
        map_width: int = 200,
        map_height: int = 200,
        origin: list=None,
        perturbation_size: float=1.,
        clean_up: bool=False,
        ground_clearance_m: float = 0.05,
        min_num_agents: int = 1,
        max_num_agents: int = 3,
        max_rock_size: float= 10., #mss
        rover_width: float= 1.0,
        rover_length: float= 0.65,
        rovers_offset: float= 0.04,
        leader_start_yaw_rad: float= math.pi/2,
        goal_yaw_rad: float= math.pi/2,
        leader_start_x_m: float= 0.,
        leader_start_y_m: float= 0.,
        goal_x_m: float= 1.,
        goal_y_m: float= 1.,
        formation_shape: int=None,
        rrt_propagation_distance: float=0.27,
        rrt_goal_sampling_bias: int=5,
        rrt_max_sample_trials: int=1000,
        rrt_max_iterations: int=10000,
        rrt_max_formation_deviation_m: float=0.1,
        rrt_formation_scaling: float=0.25,
        rrt_search_distance_multiplier: float=2.0,
        rrt_goal_capture_radius_m: float=0.2,
        verbose: bool = False,
):
    def vprint(msg):
        if verbose:
            print(msg)
    vprint("Starting Monte Carlo with seed {}".format(seed))
    if seed is not None:
        seed=int(seed)
        random.seed(seed)
    # seed = 16


    if origin is None or len(origin)==0:
        map_origin_x = -(map_width) / 2.0 * resolution;
        map_origin_y = -(map_height) / 2.0 * resolution;
        origin = [map_origin_x, map_origin_y]

    
    # Pick number of agents between 1 and 3
    num_agents = random.randrange(min_num_agents,max_num_agents+1)
    vprint("Picked n={} agents".format(num_agents))

    # Pick formation shape
    if formation_shape is None:
        formation_shape = 0
        if num_agents>2:
            formation_shape = random.randrange(0,3)
        vprint("Picked formation shape {}".format(formation_shape))



    if rovers_offset is None:
        vprint("Rovers_offset unset")
        offsets_map = {
            1:
            {
                0: 
                {
                    0: [ 0.0,  0.0,  0.]
                }
            },
            2: 
            {
                0: 
                {
                    0: [ 0.0,  0.0,  0.0],
                    1: [-0.5,  0.5,  0.0]
                }
            },
            3: 
            {
                0:
                {
                    0: [ 0.0,  0.0,  0.0 ],
                    1: [-0.5,  0.5,  0.0 ],
                    2: [-0.5, -0.5,  0.0 ]
                },
                1:
                {
                    0: [ 0.0,  0.0,  0.0],
                    1: [-0.8,  0.0,  0.0],
                    2: [ 0.8,  0.0,  0.0]
                },
                2:
                {
                    0: [ 0.0,  0.0,  0.0],
                    1: [ 0.,   0.8,  0.0],
                    2: [ 0.,  -0.8,  0.0]
                },
            },
        }
    else:
        vprint("Rovers_offset set")
        offsets_map = {
            1:
            {
                0:
                {
                    0: [ 0.0,  0.0,  0.]
                }
            },
            2: 
            {
                0:
                {
                    0: [ 0.0,  0.0,  0.0],
                    1: [0., rovers_offset,  0.0]
                },
                1:
                {
                    0: [ 0.0,  0.0,  0.0],
                    1: [-rovers_offset,  rovers_offset,  0.0]
                }
            },
            3: 
            {
                0:
                {
                    0: [ 0.0,  0.0,  0.0 ],
                    1: [-rovers_offset,  rovers_offset,  0.0 ],
                    2: [-rovers_offset, -rovers_offset,  0.0 ]
                },
                1:
                {
                    0: [ 0.0,  0.0,  0.0],
                    1: [-rovers_offset,  0.0,  0.0],
                    2: [ rovers_offset,  0.0,  0.0]
                },
                2:
                {
                    0: [ 0.0,  0.0,  0.0],
                    1: [ 0.,   rovers_offset,  0.0],
                    2: [ 0.,  -rovers_offset,  0.0]
                },
            },
        }

    assert formation_shape in offsets_map[num_agents].keys(), "ERROR: formation shape {} invalid for {} agents".format(
        formation_shape,num_agents
    )

    offsets_list = [v for k, v in offsets_map[num_agents][formation_shape].items()]

    # Pick goal location

    if goal_x_m is None:
        goal_x_m = random.uniform(-(map_width  * resolution / 2), map_width  * resolution / 2)
    if goal_y_m is None: 
        goal_y_m = random.uniform(-(map_height * resolution / 2), map_height * resolution / 2)
    if goal_yaw_rad is None:
        goal_yaw_rad = math.pi/2

    inferred_goal_locations = []
    for agent in range(num_agents):
        _goal_location = [
            goal_x_m + offsets_list[agent][0]*np.cos(goal_yaw_rad) - offsets_list[agent][1]*np.sin(goal_yaw_rad),
            goal_y_m + offsets_list[agent][0]*np.sin(goal_yaw_rad) + offsets_list[agent][1]*np.cos(goal_yaw_rad),
            (goal_yaw_rad+offsets_list[agent][2]) % math.pi
            ]
        inferred_goal_locations.append(_goal_location)

    # Pick starting location

    starting_locations = []
    if leader_start_x_m is None:
        leader_start_x_m = random.uniform(-(map_width  * resolution / 2), map_width  * resolution / 2)
    if leader_start_y_m is None:
        leader_start_y_m = random.uniform(-(map_height * resolution / 2), map_height * resolution / 2)
    if leader_start_yaw_rad is None:
        leader_start_yaw_rad = math.pi/2

    for follower in range(num_agents):
        _nominal_offset = [
            offsets_map[num_agents][formation_shape][follower][0]*np.cos(leader_start_yaw_rad) - offsets_map[num_agents][formation_shape][follower][1]*np.sin(leader_start_yaw_rad),
            offsets_map[num_agents][formation_shape][follower][0]*np.sin(leader_start_yaw_rad) + offsets_map[num_agents][formation_shape][follower][1]*np.cos(leader_start_yaw_rad),
            offsets_map[num_agents][formation_shape][follower][2]
        ]
        _start_location = [
            _nominal_offset[0]+leader_start_x_m+random.random()*float(perturbation_size),
            _nominal_offset[1]+leader_start_y_m+random.random()*float(perturbation_size),
            (_nominal_offset[2]+leader_start_yaw_rad) % math.pi
            ]
        starting_locations.append(_start_location)

    print_rover_locations_vnl("fs_start_goal_locations_{}.vnl".format(seed), [starting_locations, inferred_goal_locations])

    vprint("Picked starting and goal locations")
    # cli_caller = """
    #             ./bin/libcadre-formation-sensing-cli \
    #             -c "./share/libcadre-formation-sensing/kernels/signed_distance_field.cl.bin" \
    #             -m "./res/map.csv" \
    #             -o "0. 0. 0." -o "-0.5 0.5 0.0" -o "-0.5 -0.5 0" \
    #             -s "2. -2. 1.57" -s "1.5 -2. 1.57" -s "3. -2.2 1.57" \
    #             -g "5.25 -3 -1.57" \
    #             -w 234 -h 159 -r 0.05 \
    #             -x "-5.85 -3.975" \
    #             -t 10 \
    #             -f 0.291 \
    #             -v 0.02 \
    #             -a 0.02 \
    #             -d 0.5 \
    #             -u "_test_log"
    #             """
    
    cli_caller_list = [
        executable,
    ]

    # Map location
    cli_caller_list.append("-m")
    cli_caller_list.append("./res/map_mc_{}.csv".format(seed))

    # Map properties

    cli_caller_list.append("-w")
    cli_caller_list.append("{}".format(map_width))
    cli_caller_list.append("-h")
    cli_caller_list.append("{}".format(map_height))
    cli_caller_list.append("-r")
    cli_caller_list.append("{}".format(resolution))
    cli_caller_list.append("-x")
    cli_caller_list.append("{} {}".format(origin[0], origin[1]))

    # Start locations
    for agent in range(num_agents):
        cli_caller_list.append("-s")
        cli_caller_list.append("{} {} {}".format(
            starting_locations[agent][0],
            starting_locations[agent][1],
            starting_locations[agent][2]
            )
        )
    # Offsets
    for agent in range(num_agents):
        cli_caller_list.append("-o")
        cli_caller_list.append("{} {} {}".format(
            offsets_list[agent][0],
            offsets_list[agent][1],
            offsets_list[agent][2]
            )
        )

    # Goal
    cli_caller_list.append("-g")
    cli_caller_list.append("{} {} {}".format(
        goal_x_m,
        goal_y_m,
        goal_yaw_rad,
        )
    )

    # Rover Dimentions
    cli_caller_list.append("-k")
    cli_caller_list.append("{}".format(rover_width))
    cli_caller_list.append("-l")
    cli_caller_list.append("{}".format(rover_length))

    # Suffix
    cli_caller_list.append("-u")
    cli_caller_list.append(suffix)

    # RRT propagation distance
    cli_caller_list.append("-p")
    cli_caller_list.append("{}".format(rrt_propagation_distance))
    # RRT goal sampling bias
    cli_caller_list.append("-b")
    cli_caller_list.append("{}".format(rrt_goal_sampling_bias))
    # RRT max sample trials
    cli_caller_list.append("-e")
    cli_caller_list.append("{}".format(rrt_max_sample_trials))
    # RRT max iterations
    cli_caller_list.append("-n")
    cli_caller_list.append("{}".format(rrt_max_iterations))
    # RRT max formation deviation
    cli_caller_list.append("-j")
    cli_caller_list.append("{}".format(rrt_max_formation_deviation_m))
    # RRT formation_scaling
    cli_caller_list.append("-z")
    cli_caller_list.append("{}".format(rrt_formation_scaling))
    # RRT search distance multiplier
    cli_caller_list.append("-q")
    cli_caller_list.append("{}".format(rrt_search_distance_multiplier))
    # RRT goal capture radius
    cli_caller_list.append("-y")
    cli_caller_list.append("{}".format(rrt_goal_capture_radius_m))

    # Create a map
    vprint("Creating map...")
    cfa_level = .02
    x_limits = [0,map_width*resolution]
    y_limits = [0,map_height*resolution]
    
    map= MapGenerator(cfa=cfa_level,x_limits=x_limits,y_limits=y_limits,verbose=VERBOSE)

    # Crater specs
    crater_diameters = [0.65, 1.35]
    crater_densities = [0.15, 0.05]

    vprint("Setting locations to avoid...")
    locations_to_avoid = [[],[]]

    for agent in range(num_agents):
        # x,y = map.convert_to_grid_frame(starting_locations[agent][0], starting_locations[agent][1], map_width, map_height, resolution)
        # locations_to_avoid[0].append(x)
        # locations_to_avoid[1].append(y)
        # x,y = map.convert_to_grid_frame(goal_x_m+offsets_list[agent][0], goal_y_m+offsets_list[agent][1], map_width, map_height, resolution)
        # locations_to_avoid[0].append(x)
        # locations_to_avoid[1].append(y)
        # Start
        locations_to_avoid[0].append(starting_locations[agent][0]-origin[0])
        locations_to_avoid[1].append(starting_locations[agent][1]-origin[1])
        # And end
        locations_to_avoid[0].append(inferred_goal_locations[agent][0]-origin[0])
        locations_to_avoid[1].append(inferred_goal_locations[agent][1]-origin[1])

    locations_to_avoid_in_map_frame = map._initialize_rover_target_locations(locations_to_avoid[0], locations_to_avoid[1], 2*max(rover_width , rover_length))
    
    vprint("Generating actual map...")
    # dummy_Map = map.create_map()
    dummy_Map = map.create_terrain_map(
        cfa=cfa_level,
        seed=seed,
        use_surveyor_power_law=True, 
        ground_clearance_m=ground_clearance_m,
        crater_diameters=crater_diameters,
        crater_densities=crater_densities,
        locations_to_avoid=locations_to_avoid_in_map_frame,
        max_rock_size=max_rock_size)

    vprint("Convertingo to grid...")
    dummy_grid = dummy_Map.generate_grid_mpl(resolution=resolution, map_file_name="map_{}.png".format(seed))

    map_name = "res/map_mc_{}.csv".format(seed)
    arr = np.asarray(dummy_grid)
    pd.DataFrame(arr).to_csv(map_name, sep=',',index=False)  

    # Run!
    vprint("Invoking CLI!")
    start_time = time.time()
    # print(cli_caller_list)
    # cli_caller_str = ""
    # for _entry in cli_caller_list:
    #     cli_caller_str+=' '
    #     cli_caller_str+=_entry
    # print(cli_caller_str)

    with open("stderr{}.log".format(suffix),'w') as errorpipe:
        with open("stdout{}.log".format(suffix),'w') as errorpipe:
            subprocess.run(cli_caller_list,
                        stdout=errorpipe,
                        stderr=errorpipe)
    end_time = time.time()

    # Exit conditions
    exit_conditions = {
        "invalid_start": "Start formation is not in free space--initial conditions are invalid!",
        "invalid_goal": "The formation goal for the team is not in free space--initial conditions are invalid!",
        "invalid_tolerances": "Given the specified formation and the error bounds provided in the options, the planner cannot generate trajectories with tolerances greater than the minimum",
        "rrt_no_path": "RRT was unable to find a path close enough to goal!",
        "rrt_ok": "RRT planner successfully computed an in-formation path for the team!",
        "success": "An in-formation trajectory was successfully computed for the team!"
    }

    # Examine
    outcomes = []
    success = False
    errors = False
    nodes_in_plan=None

    with open("stderr{}.log".format(suffix),'r') as errorpipe:
        # If nonzero number of rows, report
        _errors = errorpipe.read()
        if len(_errors):
            outcomes.append("errors")
            errors = True

    with open("stdout{}.log".format(suffix),'r') as errorpipe:
        _log = errorpipe.read()
        for exit_condition, exit_condition_log in exit_conditions.items():
            if exit_condition_log in _log:
                outcomes.append(exit_condition)
            if exit_conditions["success"] in _log:
                success=True


    if success:
        with open("fs_trajectories_{}.vnl".format(seed),'r') as trajfile:
            nodes_in_plan = len(trajfile.readlines())

    # Num Agents, Exit Message, Success Bool, Steps to Reach Plan, Formation Percent, Seed
    return_stats = {
        "num_agents": num_agents,
        "formation_shape": formation_shape,
        "outcomes": tuple(outcomes),
        "success": success,
        "errors": errors,
        "nodes_in_plan": nodes_in_plan,
        "seed": seed,
        "wall_time_s": end_time-start_time if success else np.nan
    }

    if clean_up and len(outcomes):
        created_files = [
            "./res/map_mc_{}.csv".format(seed),
            "fs_trajectories_{}.vnl".format(seed),
            "fs_start_goal_locations_{}.vnl".format(seed),
            "fs_starting_locations_{}.vnl".format(seed),
            "fs_goal_locations_{}.vnl".format(seed),
            "fs_pre_shortcutting_{}.vnl".format(seed),
            "fs_pre_smoothing_{}.vnl".format(seed),
            "fs_post_smoothing_{}.vnl".format(seed),
            "fs_trajectories_{}.vnl".format(seed),
            "fs_map_{}.pbm".format(seed),
            "fs_sdf_{}.pbm".format(seed),
            "fs_sdf_{}.pgm".format(seed),
            "stderr_{}.log".format(seed),
            "stdout_{}.log".format(seed),

        ]
        for created_file in created_files:
            try:
                pathlib.Path.unlink(pathlib.Path(created_file))
            except FileNotFoundError:
                pass
    return return_stats

def chunk_handler(i):
    return run_one_monte_carlo(*i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot trajectories generated by formation-sensing.')
    parser.add_argument('-n','--number_trials', help='The number of trials.', type=int, default=100)
    parser.add_argument('-e','--executable', help='The path to the CLI executable.', default='../build/bin/libcadre-formation-sensing-cli')
    parser.add_argument('-r','--resolution', nargs='?', help='The map resolution.', type=float, default=0.05)
    parser.add_argument('-w','--map_width', nargs='?', help='The map width in px.', type=float, default=500)
    parser.add_argument('-i','--map_height', nargs='?', help='The map resolution.', type=float, default=500)
    parser.add_argument('-x','--origin', help='Set the origin of the map.', default=None)
    parser.add_argument('-p','--perturbation_size', help='How perturbed are the starts.', default=0.)
    parser.add_argument('-g','--ground_clearance', help='The rovers\' ground clearance, in m.', default=0.05)
    parser.add_argument('-m','--min_num_agents', help='The minimum number of agents', default=1)
    parser.add_argument('-a','--max_num_agents', help='The maximum number of agents', default=3)
    parser.add_argument('-b','--max_rock_size', help='Max rock size in m', default=30)
    parser.add_argument('-c','--cleanup', help='Clean up after execution.', action='store_true')
    parser.add_argument('-k','--rover_width', help='Rover width in m.', type=float, default=0.9) # mss
    parser.add_argument('-l','--rover_length', help='Rover length in m.',type=float, default=0.65)
    parser.add_argument('-f','--rovers_offset', help='Rovers offset in m.',type=float, default=4.0)
    parser.add_argument('-y','--leader_start_yaw_rad', help='Leader start yaw in rad.',type=float, default=math.pi/2)
    parser.add_argument('-o','--goal_yaw_rad', help='Goal yaw in rad.',type=float, default=math.pi/2)
    parser.add_argument('-u','--leader_start_x_m', help='Leader start x in m.',type=float, default=-19)
    parser.add_argument('-v','--leader_start_y_m', help='Leader start y in m.',type=float, default=0)
    parser.add_argument('-s','--goal_x_m', help='Goal x in m.',type=float, default=19)
    parser.add_argument('-t','--goal_y_m', help='Goal y in m.',type=float, default=5)
    parser.add_argument('-pr','--rrt_propagation_distance', help='RRT propagation distance',type=float, default=0.27)
    parser.add_argument('-bi','--rrt_goal_sampling_bias', help='',type=int, default=5)
    parser.add_argument('-est','--rrt_max_sample_trials', help='',type=int, default=1000)
    parser.add_argument('-nit','--rrt_max_iterations', help='',type=int, default=10000)
    parser.add_argument('-j','--rrt_max_formation_deviation_m', help='',type=float, default=0.1)
    parser.add_argument('-z','--rrt_formation_scaling', help='',type=float, default=0.25)
    parser.add_argument('-q','--rrt_search_distance_multiplier', help='',type=float, default=2.0)
    parser.add_argument('-yradius','--rrt_goal_capture_radius', help='',type=float, default=0.2)
    parser.add_argument('--formation_shape', help='',type=int, default=0)


    parser.set_defaults(cleanup=False)

    args = parser.parse_args()

    # print(args)

    if args.origin is not None:
        origin = [float(f) for f in args.origin.split(" ")]
    else:
        origin = args.origin # Just keep it None

    # mc_dataframe = pd.DataFrame()
    with Pool() as pool:
        inputs = [(
                args.executable,               # Executable
                _seed,                         # Seed
                "_{}".format(_seed),           # Suffix
                float(args.resolution),         # Resolution
                float(args.map_width),          # width
                float(args.map_height),         # height
                origin,                        # Map origin
                float(args.perturbation_size),  # Start location perturbation size,
                args.cleanup,                         # Should we clean up
                float(args.ground_clearance),   # Ground clearance,
                int(args.min_num_agents),           # Min number of agents,
                int(args.max_num_agents),           # Max number of agents,
                float(args.max_rock_size),          # Max rock size,
                float(args.rover_width),            # Rover width in m
                float(args.rover_length),           # Rover length in m
                float(args.rovers_offset),          # Rovers offset in m
                float(args.leader_start_yaw_rad),   # Leader start yaw
                float(args.goal_yaw_rad),           # Goal yaw in rad
                args.leader_start_x_m,              # Leader start x in m
                args.leader_start_y_m,              # Leader start y in m
                args.goal_x_m,                      # Leader goal x in m
                args.goal_y_m,                      # Leader goal y in m
                # float(args.max_rock_diameter),      # Leader goal y in m
                args.formation_shape,                # Formation shape
                args.rrt_propagation_distance,
                args.rrt_goal_sampling_bias,
                args.rrt_max_sample_trials,
                args.rrt_max_iterations,
                args.rrt_max_formation_deviation_m,
                args.rrt_formation_scaling,
                args.rrt_search_distance_multiplier,
                args.rrt_goal_capture_radius,
                VERBOSE,
            )
             for _seed in range(args.number_trials)]
        if PARALLEL:
            many_return_stats = list(tqdm(pool.imap(chunk_handler, inputs), total=args.number_trials))
        else:
            many_return_stats = []
            for input in inputs:
                many_return_stats.append(chunk_handler(input))
        
    mc_dataframe = pd.DataFrame.from_records(many_return_stats)

    print(mc_dataframe)

    mc_dataframe.to_csv("MonteCarlo_{}.results".format(args.number_trials), sep=';',index=False)  
    mc_dataframe.to_excel("MonteCarlo_{}.xlsx".format(args.number_trials))

    stats_errors = mc_dataframe.errors.value_counts().to_dict()
    stats_agents = mc_dataframe.num_agents.value_counts().to_dict()
    stats_successes = mc_dataframe.success.value_counts().to_dict()
    # stats_nodes_in_plan = mc_dataframe.nodes_in_plan.value_counts().to_dict()
    stats_outcomes = mc_dataframe.outcomes.value_counts().to_dict()

    num_errors = 0 if (True not in stats_errors.keys()) else stats_errors[True]
    # print(stats_agents, type(stats_agents))
    num_successful = 0 if (True not in stats_successes.keys()) else stats_successes[True]
    # print(stats_nodes_in_plan, type(stats_nodes_in_plan))
    print(stats_outcomes, type(stats_outcomes))

    print("We examined {} Monte Carlo trials.\n".format(len(mc_dataframe)),
          "{} ({}%) were successful.\n".format(num_successful, 100*num_successful/len(mc_dataframe)),
          "Errors were reported in {} ({}%) runs.\n".format(num_errors, 100*num_errors/len(mc_dataframe)),
          "The median, min, and max number of nodes were {}, {}, {}. The stddev was {}.\n".format(
              mc_dataframe.nodes_in_plan.median(),
              mc_dataframe.nodes_in_plan.min(),
              mc_dataframe.nodes_in_plan.max(),
              mc_dataframe.nodes_in_plan.std(),
              ),
          "The median, min, and max time were {}, {}, {} s. The stddev was {} s.\n".format(
              mc_dataframe.wall_time_s.median(),
              mc_dataframe.wall_time_s.min(),
              mc_dataframe.wall_time_s.max(),
              mc_dataframe.wall_time_s.std(),
              ),
          "There were {} runs ({}%) with 1 agent, {} runs ({}%) with 2 agents, and {} runs ({}%) with 3 agents.".format(
              stats_agents.get(1, 0), 100*stats_agents.get(1, 0)/len(mc_dataframe),
              stats_agents.get(2, 0), 100*stats_agents.get(2, 0)/len(mc_dataframe),
              stats_agents.get(3, 0), 100*stats_agents.get(3, 0)/len(mc_dataframe),
              ),
          "The outcomes we saw were:\n",
          ["{}: {} ({}%)".format(k, v, 100*v/len(mc_dataframe)) for k, v in stats_outcomes.items()]
          )

