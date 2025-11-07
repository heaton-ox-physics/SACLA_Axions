# SACLA_Axions
Scripts for SACLA beamtime.

## Usage of read_det_mpi in mpccd.py:

1. request resources  (16 cores, 100GB memory)
   qsub -I -l select=1:ncpus=16:mem=100gb
2. load module
   module load python/SACLA_python-3.7
3. run following scrip using mpi (mpiexec -n 5 python3 test_mpccd.py)
   test_mpccd.py:

   from mpccd import MPCCDProcessing
   MP=MPCCDProcessing('/home/sifei/', 'MPCCD-1B1-M03-006')
   data = MP.read_det_mpi(1616893, False)
   
