# Lunar Rover Simulator

This repository contains a simulator for Lunar communications in rovers exploration. In this simulation, rovers explore a lunar surface, generate digital elevation maps (DEMs) as packets as they explore and from objectives. Rovers test several communication protocols for enhancing reliability and robustness of communications.

![Demo Video](Video/Demo.gif)
> Multi-agent lunar simulator. Gif generated with https://ezgif.com/video-to-gif

<!-- ## Path planner -->

<!-- - Docker setup (https://docs.docker.com/get-started/get-docker/)
    - pull ubuntu image
    - set the docker file to install dependencies and packets
    - Build the image
        - open 6080 port
            - docker run -d -p 8080:80 docker/welcome-to-docker
        - sudo apt install and re build images
    - save image in docker descktop and just load it next times. automatic after the build
    - run docker container with built image
    - clone repos inside and stuff
    - cmake ..
    - cmake --build . or make -->

## Environment
- Random rocks are generated emulating the lunar surface
- Rovers generate random objectives in the environment
- Rovers stop for several time steps once they reach an objective
- Packet-level simulation enabled, rovers share real packets among them. Each rover is a source node, and the lander is the sink node for all rovers

## Map generation
- Random rocks (simple map)
- Realistic maps based on Lunar statistical distribution. Craters and rocks are generated

## Path and objectives planning
1. Simple scenario:
    - Rovers set random objectives among the surface, and move towards them following a path created with A* algorithm
    - Some noise is added to rovers' movement to enhance realism
2. PSE path and objective planner:
    - Rovers set their own objectives
    - Path generated as they discover it
    - Trajectories are normalized 0-1 instead of meters

## Packet generation
- Rovers generate packets as they move, every n time steps
- Rovers generate n packets once they reach an objective

## Exploration
Each packet is associated with coordinates in the map. The area covered by the packets that reached the lander is green. The area covered by rovers will remain red until the associated generated packet reaches the lander.

## Communications
- Queuing:
    - Every rover makes a different queue per neighbor rover (if not directly connected to the lander, in that case packets are simply going there)
    - Every time step, the first n packets are assigned a queue
    - Every time step, a set of packets go to the next hop depending on the tx rate
    - Every queue follows FIFO policy
    - **Drop packet policy:**
        - If rover is full, drops oldest packets and appends the new generated ones.
        - Rovers do not send packets to a neighbor that is full. 
        - If a full rover receives a packet from a neighbor, which should not happen, the rover will drop its oldest packet and append the new received one.

- Policies:
    - Greedy policy:
        - Sends to lander if direct link
        - Sends to neighbor rovers if they are indirectly connected to the lander. Specifically to the neighbor rover that is the closest to the lander in number of hops
        - If a neighbor is full, it won't attempt to send packets there. Possible issues: Full rovers are used to compute shortest paths. Full nodes could be excluded from local graphs, but I do not want to increase computational complexity. Should not be a problem, but worth to keep an eye on this.
        - Stores otherwise
    - Spray and wait:
        - Sends to lander if direct link
        - Makes L copies of a packet and spread them
        - Packet copy_counter set to L that indicates the number of copies that can be done to a packet.
        - If L>1:
            - Sends copy to neighbor. New packet copy_counter: floor(L/2). Original packet copy_counter: L-floor(L/2). This is the binary split, optimal when nodes movement is IID.
        - They won't send a packet to a rover that is either full or already has that packet
        - Future enhancements for Spray and Wait:
            - Spray only to neighbors more likely to deliver the packet
            - Leverage multi-hop direct routes (this assumes graph knowledge)
            - Add packet removal policy:
                - Receive ack? This is hard since nonnections are very dynamic
                - Time To Live (TTL)
                - N copies
    - Multi-Agent Deep Reinforcement Learning (MA-DRL)
    Each rover is a different agent that makes local observations. Rovers' experiences feed a centralized neural network, where there is centralized training and decentralized execution.
        - State Space:
            - B: Buffer level
            - LC: Lander connection, binary
            - TTL: Time-To-Lander
            - N-Neighbors [B, LC, TTL]
        - Action Space:
            - 0: Store
            - 1-N: Forward to neighbor n
        - Rewards
            - Deliver: Delta
            - Load Balance: Negative exponential based on buffer usage

## Multi-agent_autonomy
It could be interesting to represent the current state with a Graph Attention Network, where each node has information about its current load something related to their future:
- How close they are to their next objective
- Are they going to be close to the sink node (lander) in their way to their objective?
- Expected time to sleep (inactive node)

## Usage
    python lunar_simulator.py

## Installation

To set up the environment and install all required packages, follow these steps (**Python3.9.6**):

1. **Clone the repository**:
    ```sh
    git clone https://fornat1.jpl.nasa.gov/fedeloz/multi-agent_autonomy.git
    cd /path/to/multi-agent_autonomy
    ```

2. **Create a virtual environment** (optional but recommended):
    ```sh
    python -m venv venv
    source .venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Path planner Docker installation (VPN and Docker required)

1. **Pull environment** and **install dependencies**
    ```sh
    sh setup.sh
    ```

2. Create the **docker image** (Takes about 500s)
    ```sh
    docker build -t path-planner-image . --no-cache 
    ```

3. Run a **container** with the docker image
    ```sh
    docker run -it -v $(pwd) path-planner-image bash  
    ```


## Current conclusions
- The current exploration algorithmic gives very low contacts among rovers since each of them goes to a different area
- It makes sense to drop the oldest generated packets to let the new ones get in in DTN:
    - You might have sent a copy of the oldest to a different rover who might have already delivered it
    - In Spray and Wait specifically iy is really hard to get rid of the inactive packets; you will preserve them until you reach the destination directly, which is very inefficient.

## #ToDo:
- Mid-term ideas:
    - Make first tests with Graph Attention Networks
    - The GAT should have information about the neighbor congestion and if it is gonna pass close to the lander when going to the next objective for how long and how sure. this both could just be a confidence number like an offloading confidence number
    - Add mobility options in the state space:
        - Go back to coverage area
        - Do not move (useful when downloading data)

## known issues
- PSE path planner gets stuck when the Field of View (FoV) is not 360 deg.
- When increase the movement step (step_size) they can get stuck with obstacles
- Spray and Wait: When checks if a node already has a packet, only checks the current buffer and does not keep track of the history of packets. This is done because the computational complexity would increase with time, and the effect over the algorithm is minimal; if a node had a packet in the past and not anymore it means that the packet was delivered due to the nature of Spray and Wait, so receiving again the same packet will not affect the effective throughput (which accounts for single-copy delivered packets)
- When the trajectories and goals are saved at the random scenario (not PSE path planner), and then loaded, the goals will not appear in the animation. Probably this is because they are stored worngly in .timed_goals.append(...) inside generate_objective(). However this is not very relevant currently since usually the trajectories loaded are the ones generated by PSE, which works properly. **In short:** You can only import trajectories generated by PSE, not random objectives.