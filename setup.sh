#!/bin/sh

# 1) Clone main repo with submodules
git clone --recurse-submodules https://cadregitlab.jpl.nasa.gov/cadre/cadre-autonomy-pse/cadre-pse.git
cd cadre-pse || exit 1

# 2) Remove mexec references
sed -i '' '/mexec/d' .gitmodules
sed -i '' '/mexec/d' CMakeLists.txt

# 3) Deinit and update submodules
git submodule deinit -f .
git submodule update --init --recursive

# 4) Clone nanoflann and checkout specific commit
git clone https://github.com/jlblancoc/nanoflann.git
cd nanoflann
git reset --hard 3900193f7e200c9e152aa4b240d7eca47a830a80
cd ..

# 5) Clone ETL and checkout specific commit
git clone https://github.com/ETLCPP/etl.git
cd etl
git reset --hard 0676ded8cfc3b957f15a0af2c142b1623d378705
cd ..

# 6) Clone doctest
git clone https://github.com/doctest/doctest.git doctest

# 7) Checkout specific branch in agent-exploration
cd agent-exploration
git checkout dev/20-dijstra-set-bug
cd ..

# 8) Clone and checkout specific branch in simple-sim
git clone https://cadregitlab.jpl.nasa.gov/cadre/cadre-autonomy-pse/simple-sim.git
cd simple-sim
git checkout dev/exploration-simple-sim-fr
cd ..

# 10) Modify code (for OpenCL compability)
# Modify ComputableProgram.hpp: insert define at line 32
sed -i '' '32i\
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY
' ./formation-sensing/src/cadre/opencl/ComputableProgram.hpp

# Modify program_factory.cpp: comment out line 87
sed -i '' '87s/^/\/\/ /' ./formation-sensing/src/cadre/opencl/program_factory.cpp
