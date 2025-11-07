#### We want a python code that takes a run and makes an h5 file for the CITIUS detector
import os 
import sys
import numpy as np
# import matplotlib.pyplot as plt
# import stpy
import dbpy
# import stpy
import h5py
from mpi4py import MPI
### Is the CITIUS package installed?
import ctdapy_xfel
import time

def displayErrorMessage(rank):
    singlePrint(rank, "citiusToH5 uses ctdapy_xfel to convert CITIUS data to h5 files \n")
    singlePrint(rank, "Usage: mpiexec -n <nProcs> python citiusToH5.py -dir <writeableDirectory=.> -r runNumber -BL <beamline=3> ~ for a single run")
    singlePrint(rank, "Usage: mpiexec -n <nProcs> python citiusToH5.py -dir <writeableDirectory=.> -rLow <runNumberLow> -rHigh <runNumberHigh> -BL <beamline=3> ~ for a series of runs\n")
    singlePrint(rank, "Program will sort for runs with CITIUS")
    

def singlePrint(rank,message):
    if (rank == 0):
        print(message)
    else:
        pass

def getPulseEnegyInJ(beamLine,run,trainID):
    ### Replace with proper function
    return 1

# which frames each rank owns (same frames_for_rank as above)
def frames_for_rank(rank, size, N):
    base = N // size
    rem = N % size
    start = rank * base + min(rank, rem)
    stop  = start + base + (1 if rank < rem else 0) + base - base  # same formula; simpler in final code
    # simpler:
    start = rank * (N // size) + min(rank, N % size)
    stop = start + (N // size) + (1 if rank < (N % size) else 0)
    return start, stop

def main():

    MPI_WORLD   = MPI.COMM_WORLD
    nProcs      = MPI_WORLD.Get_size()
    rank        = MPI_WORLD.Get_rank()

    ### Default values
    runNumber = -1
    lowRunNumber = -1
    highRunNumber = -1
    beamLine = 3
    runNumbers = []
    writeableDirectory = "."

    upperThreshold = None
    lowerThreshold = -10
    
    ### Read in user prompts
    for i in range(np.shape(sys.argv)[0]):
        if ( sys.argv[i] == "-h" ) or ( sys.argv[i] == "-help" ) or ( sys.argv[i] == "--help" ):
            displayErrorMessage(rank)
    for i in range(np.shape(sys.argv)[0]):
        if sys.argv[i] == "-rLow":
            lowRunNumber = int(sys.argv[i+1])
    for i in range(np.shape(sys.argv)[0]):
        if sys.argv[i] == "-rHigh":
            highRunNumber = int(sys.argv[i+1])

    for i in range(np.shape(sys.argv)[0]):
        if sys.argv[i] == "-r":
            runNumber = int(sys.argv[i+1])
            
    for i in range(np.shape(sys.argv)[0]):
        if sys.argv[i] == "-BL":
            beamLine = int(sys.argv[i+1])

    for i in range(np.shape(sys.argv)[0]):
            if sys.argv[i] == "-dir":
                writeableDirectory = str(sys.argv[i+1])

    for i in range(np.shape(sys.argv)[0]):
            if sys.argv[i] == "-upperThreshold":
                upperThreshold = str(sys.argv[i+1])

    for i in range(np.shape(sys.argv)[0]):
            if sys.argv[i] == "-lowerThreshold":
                lowerThreshold = str(sys.argv[i+1])
            
        
    if ((runNumber < 0 and lowRunNumber < 0 and highRunNumber < 0) or (runNumber < 0 and (lowRunNumber < 0 or highRunNumber < 0))):
        singlePrint(rank,"Error: No run numbers given")
        singlePrint(rank,"Run with --help for assistance")
        sys.exit(1)

    if (lowRunNumber > 0 and highRunNumber > 0):
        for i in range(lowRunNumber,highRunNumber+1):
            runNumbers.append(i)
    
    if (runNumber > 0):
        if (runNumber in runNumbers):
            pass
        else:
            runNumbers.append(runNumber)
    ### Filter by runs with CITIUS
    runsToConsider = []
    for run in runNumbers:
        if (ctdapy_xfel.get_runstatus(beamLine,run) == 0):
            runsToConsider.append(run)

    runNumbers = runsToConsider
        
    
    outString = "Runs to process:  " + str(runNumbers) + "\n"
    singlePrint(rank,outString)

    ### CONSTANTS
    CITIUS_IMAGE_WIDTH  = 728
    CITIUS_IMAGE_HEIGHT = 384

    x1, y1, x2, y2 = 160, 340, 230, 400  # ROI coordinates 
    
    for run in runNumbers:
        ### Get buffer
        t0 = time.time()
        buffer = ctdapy_xfel.CtrlBuffer(beamLine,run)
        sensorID = buffer.read_detidlist()[0]
        
        CITIUS_EMPTY_MASK = np.zeros((CITIUS_IMAGE_WIDTH,CITIUS_IMAGE_HEIGHT),dtype=np.uint8)
        
        buffer.read_badpixel_mask(CITIUS_EMPTY_MASK,0)
        mask   = CITIUS_EMPTY_MASK
        tagList = buffer.read_taglist()
        numberOfTrains = len(tagList)

        singlePrint(rank, "Run number      : " + str(run))
        singlePrint(rank, "Number of trains: " + str(numberOfTrains))


        summedArray = np.zeros(np.shape(CITIUS_EMPTY_MASK))
        summedArrayNoNorm = np.zeros(np.shape(CITIUS_EMPTY_MASK))
        count,bins = np.histogram(CITIUS_EMPTY_MASK,bins=np.arange(0,100,0.05),density=False)
        allCounts = np.zeros(np.shape(count))

        localSummedArray       = np.zeros(np.shape(CITIUS_EMPTY_MASK))
        localSummedArrayNoNorm = np.zeros(np.shape(CITIUS_EMPTY_MASK))
        count,bins = np.histogram(CITIUS_EMPTY_MASK,bins=np.arange(0,100,0.05),density=False)
        localAllCounts = np.zeros(np.shape(count))
        
        localSummedArray_roi   = np.zeros(np.shape(CITIUS_EMPTY_MASK[y1:y2, x1:x2]))
        count_roi,bins_roi = np.histogram(CITIUS_EMPTY_MASK[y1:y2, x1:x2],bins=np.arange(0,100,0.05),density=False)
        localAllCounts_roi = np.zeros(np.shape(count_roi))

        nTrainsLocal = 0
        
        #### It'll be more efficient to parralise over trainIds as NTrains >> NRuns

        start, stop  = frames_for_rank(rank,nProcs,numberOfTrains)
        localTagList = tagList[start:stop] 
        for train in localTagList:
            I0 = getPulseEnegyInJ(beamLine,run,train)
            CITIUS_EMPTY_ARRAY  = np.zeros((CITIUS_IMAGE_WIDTH,CITIUS_IMAGE_HEIGHT),dtype=np.float32)
            buffer.read_image(CITIUS_EMPTY_ARRAY,0,train)
            citiusData = CITIUS_EMPTY_ARRAY

            ### Apply threshold 
            if (upperThreshold != None):
                citiusData[np.where(citiusData > upperThreshold)] = np.nan
            citiusData[np.where(citiusData < lowerThreshold)] = 0.0
            ### Mask hot pixels
            citiusData[np.where(mask == 1)] = np.nan

            localSummedArray = localSummedArray + citiusData / I0
            localSummedArrayNoNorm = localSummedArrayNoNorm + citiusData
            
            count,bins = np.histogram(citiusData,bins=np.arange(0,100,0.05),density=False)
            localAllCounts = localAllCounts + count

            # apply roi and create seperate histogram
            citiusData_roi = citiusData[y1:y2, x1:x2]
            localSummedArray_roi = localSummedArray_roi + citiusData_roi / I0
            count_roi,bins_roi = np.histogram(citiusData_roi,bins=np.arange(0,100,0.05),density=False)
            localAllCounts_roi = localAllCounts_roi + count_roi
            
            nTrainsLocal = nTrainsLocal + 1

        summedArray       = MPI_WORLD.reduce(localSummedArray      , op=MPI.SUM, root=0)
        summedArray_roi   = MPI_WORLD.reduce(localSummedArray_roi  , op=MPI.SUM, root=0)
        
        summedArrayNoNorm = MPI_WORLD.reduce(localSummedArrayNoNorm, op=MPI.SUM, root=0)
        
        allCounts         = MPI_WORLD.reduce(localAllCounts        , op=MPI.SUM, root=0)
        allCounts_roi     = MPI_WORLD.reduce(localAllCounts_roi    , op=MPI.SUM, root=0)
        
        if (rank == 0):
            #
            # Saving arrays
            summedArray = summedArray / float(numberOfTrains)
            saveFileName = str(writeableDirectory) + "/citius_BL"+ str(beamLine) + "_r" + str(run) + ".npy" 
            np.save(saveFileName,summedArray)

            summedArray_roi = summedArray_roi / float(numberOfTrains)
            saveFileName = str(writeableDirectory) + "/citius_roi_BL"+ str(beamLine) + "_r" + str(run) + ".npy" 
            np.save(saveFileName,summedArray_roi)
            #
            # Saving non-normalised arrays
            saveFileName = str(writeableDirectory) + "/citiusNoNormilisation_BL"+ str(beamLine) + "_r" + str(run) + ".npy" 
            np.save(saveFileName,summedArrayNoNorm)
            #
            # Saving histograms
            binCenters = []
            for i in range(np.shape(bins)[0]-1):
                binCenters.append((bins[i]+bins[i+1]) * 0.5)
            
            saveFileName = str(writeableDirectory) + "/citiusHistogram_BL"+ str(beamLine) + "_r" + str(run) + ".npy" 
            np.save(saveFileName,np.column_stack((binCenters, allCounts)))

            binCenters_roi = []
            for i in range(np.shape(bins_roi)[0]-1):
                binCenters_roi.append((bins_roi[i]+bins_roi[i+1]) * 0.5 )
            
            saveFileName = str(writeableDirectory) + "/citiusHistogram_roi_BL"+ str(beamLine) + "_r" + str(run) + ".npy" 
            np.save(saveFileName,np.column_stack((binCenters_roi, allCounts_roi)))
            #
            #
            t1 = time.time()

            totalTime = t1-t0
            outString = str(numberOfTrains) + " trains processed in " + str(round(totalTime,5)) + "s on " +str(nProcs) +" ranks.\n" 
            singlePrint(rank,outString)
            
    sys.exit(0)


if __name__ == "__main__":
    main()
    
    
    
