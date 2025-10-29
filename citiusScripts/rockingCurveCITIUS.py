#### We want a python code that takes a run and takes the integral in an roi for the CITIUS detector
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
    singlePrint(rank, "citiusRockingCurve.py uses ctdapy_xfel to make a rocking curve measurement \n")
    singlePrint(rank, "Usage: mpiexec -n <nProcs> python citiusIntegralInRIO.py -dir <writeableDirectory=.> -r runNumber -BL <beamline=3> ~ for a single run")
    singlePrint(rank, "Program only works for one run at a time. Only the first specified run will be considered.")
    

def singlePrint(rank,message):
    if (rank == 0):
        print(message)
    else:
        pass

def getPulseEnegyInJ(beamLine,run,trainID):
    ### Replace with proper function
    return 1

def getMotorPositions(beamline,run):
    # taglist = dbpy.read_taglist_byrun(beamline, run)
    # high_tag = dbpy.read_hightagnumber(beamline, run)

    # #dbpy.read_syncdatalist_float(equipID, high_tag, taglist)
    # #output: tuple of syncDB values
    # equipID = "motorName"
    # motorPosition = np.array(dbpy.read_syncdatalist_float(equipID, high_tag, taglist))
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
    beamLine = 3
    writeableDirectory = "."

    upperThreshold = None
    lowerThreshold = -10
    ### CONSTANTS
    CITIUS_IMAGE_WIDTH  = 384
    CITIUS_IMAGE_HEIGHT = 728
    roi_LX = 0
    roi_RX = CITIUS_IMAGE_WIDTH
    roi_LY = 0
    roi_UY = CITIUS_IMAGE_HEIGHT

    
    ### Read in user prompts
    for i in range(np.shape(sys.argv)[0]):
        if ( sys.argv[i] == "-h" ) or ( sys.argv[i] == "-help" ) or ( sys.argv[i] == "--help" ):
            displayErrorMessage(rank)

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

    ### Get corners from ROI

    for i in range(np.shape(sys.argv)[0]):
            if sys.argv[i] == "-roiLx":
                roi_LX = str(sys.argv[i+1])

    for i in range(np.shape(sys.argv)[0]):
            if sys.argv[i] == "-roiRx":
                roi_RX = str(sys.argv[i+1])

    for i in range(np.shape(sys.argv)[0]):
            if sys.argv[i] == "-roiLy":
                roi_LY = str(sys.argv[i+1])

    for i in range(np.shape(sys.argv)[0]):
            if sys.argv[i] == "-roiUy":
                roi_UY = str(sys.argv[i+1])



    if ((runNumber < 0 )):
        singlePrint(rank,"Error: No run numbers given")
        singlePrint(rank,"Run with --help for assistance")
        sys.exit(1)
        
    
    outString = "Run to process:  " + str(runNumber) + "\n"
    singlePrint(rank,outString)

    
    ### Get buffer
    t0 = time.time()
    buffer = ctdapy_xfel.CtrlBuffer(beamLine,runNumber)
    sensorID = buffer.read_detidlist()[0]
    CITIUS_EMPTY_MASK = np.zeros((CITIUS_IMAGE_WIDTH,CITIUS_IMAGE_HEIGHT),dtype=np.uint8)
    buffer.read_badpixel_mask(CITIUS_EMPTY_MASK,0)
    mask   = CITIUS_EMPTY_MASK
    tagList = buffer.read_taglist()
    numberOfTrains = len(tagList)

    singlePrint(rank, "Run number      : " + str(runNumber))
    singlePrint(rank, "Number of trains: " + str(numberOfTrains))


    summedArray            = np.zeros(np.shape(CITIUS_EMPTY_MASK))
    localSummedArray       = np.zeros(np.shape(CITIUS_EMPTY_MASK))

    

    
    #### It'll be more efficient to parralise over trainIds as NTrains >> NRuns

    start, stop  = frames_for_rank(rank,nProcs,numberOfTrains)
    localTagList = tagList[start:stop] 


    ### Not implemented yet. For now, we set the motor positions to have some monotonically increasing array
    motorPositions = []
    for i in range(numberOfTrains):
        motorPositions.append(i)
    localMotorPositions = motorPositions[start:stop]
    localIntegratedIntensity = []
    jLocal = 0
    
    for train in localTagList:
        I0 = getPulseEnegyInJ(beamLine,runNumber,train)
        ### Convert I0 to number of incident photons .... TODO
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
        ### Save data as a tuple of motor position and integrated intensity
        motorPosition = localMotorPositions[jLocal]
        localIntegratedIntensity.append((motorPosition,np.nanmean(citiusData[roi_LX:roi_RX,roi_LY:roi_UY])))
        jLocal = jLocal + 1
        
    if (nProcs > 1):
        # outString ="I'm on rank " + str(rank) + ". My summed array has shape " + str(np.shape(localSummedArray))
        print(outString) 
        summedArray       = MPI_WORLD.reduce(localSummedArray        , op = MPI.SUM, root = 0)
        totalRockingCurve = MPI_WORLD.gather(localIntegratedIntensity, root = 0)
    else:
        summedArray = localSummedArray
        totalRockingCurve = localIntegratedIntensity
   

    

    if (rank == 0):
        # Turn off interactive mode
        # plt.ioff()
        # plt.imshow(summedArray,vmin=np.nanmin(summedArray), vmax = np.nanmax(summedArray),extent=[0,CITIUS_IMAGE_WIDTH,0,CITIUS_IMAGE_HEIGHT],cmap="plasma")
        # plt.scatter(roi_LX,roi_LY,color='white')
        # plt.scatter(roi_RX,roi_UY,color='white')
        # plt.scatter(roi_LX,roi_LY,color='white')
        # plt.scatter(roi_RX,roi_LY,color='white')
        # saveFileName = str(writeableDirectory) + "/citiusRockingCurve_BL"+ str(beamLine) + "_r" + str(runNumber) + ".png" 
        # plt.savefig(saveFileName,bbox_inches="tight")
        summedArray       = summedArray / float(numberOfTrains)
        saveFileName = str(writeableDirectory) + "/citiusRockingCurve_BL"+ str(beamLine) + "_r" + str(runNumber) + ".npy" 
        np.save(saveFileName,totalRockingCurve)

        saveFileName = str(writeableDirectory) + "/citius_BL"+ str(beamLine) + "_r" + str(runNumber) + ".npy" 
        np.save(saveFileName,summedArray)

        t1 = time.time()

        totalTime = t1-t0
        outString = str(numberOfTrains) + " trains processed in " + str(round(totalTime,5)) + "s on " +str(nProcs) +" ranks.\n" 
        singlePrint(rank,outString)
        
    sys.exit(0)


if __name__ == "__main__":
    main()
    
    
    
