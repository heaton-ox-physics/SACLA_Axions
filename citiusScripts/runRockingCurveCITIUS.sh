#!/bin/bash
#PBS -N citius                     
# Job name
#PBS -o output.txt                 
# Standard output file
#PBS -e error.txt                  
# Standard error file
#PBS -l walltime=01:00:00          
# Time limit (hh:mm:ss)
#PBS -l nodes=1:ppn=16
# 1 node, 4 processors per node
#PBS -V                            
# Export environment variables to the job

# Change to the directory from which the job was submitted
cd $PBS_O_WORKDIR

# Load necessary modules
module load python/SACLA_python-3.7

rm output.txt error.txt

# Run your application
time mpiexec -n ${NCPUS} python3 rockingCurveCITIUS.py -r 255550 -BL 2 
