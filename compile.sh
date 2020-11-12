#!/bin/bash
#
# Submit job as (build defaults to Release):
#
#   sbatch compile.sh
#   sbatch --export=BUILD='Debug',ALL compile.sh 
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mem=8GB

# Load modules
module purge

module purge
module load intel/17.0.1
module load gcc/6.3.0
module load cmake/gnu/3.11.4

export CC=icc
export CXX=icpc

export CMAKE_INCLUDE_PATH=$(env | grep _INC= | cut -d= -f2 | xargs | sed -e 's/ /:/g')
export CMAKE_LIBRARY_PATH=$(env | grep _LIB= | cut -d= -f2 | xargs | sed -e 's/ /:/g')

cmake ..

make
