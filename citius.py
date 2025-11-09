import os
import numpy as np
import dbpy
import sys
import matplotlib.pyplot as plt
from math import *
import stpy
import ippy
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

#### We want a python code that takes a run and makes an h5 file for the CITIUS detector
# import matplotlib.pyplot as plt
# import stpy
import dbpy
# import stpy
import h5py
from mpi4py import MPI
### Is the CITIUS package installed?
import ctdapy_xfel
import time
from scipy import ndimage


def getPulseEnegyInJ(beamLine,run,trainID):
    ### Replace with proper function
    return 1

class CITIUSReader():
    def __init__(self, read_path, bl=3, ROI=None):
        self.bl = bl
        self.read_path = read_path
        if ROI is not None: 
            self.x1, self.y1, self.x2, self.y2 = ROI[0], ROI[1], ROI[2], ROI[3]  # ROI coordinates 
            
        self.slope = 69.2 # predetermined from calibration
        
    def read_runlist(self, run_list):
        #
        img_list, imgROI_list, bins_list, counts_list, binsROI_list, countsROI_list = [[] for i in range(6)]
        #
        for run in run_list:
            fileName_img = os.path.join(self.read_path, f'citius_BL{self.bl}_r{run}.npy')
            img = np.load(fileName_img)
            img_list.append(img)

            try:
                fileName_imgROI = os.path.join(self.read_path, f'citius_roi_BL{self.bl}_r{run}.npy')
                imgROI = np.load(fileName_imgROI)
                imgROI_list.append(imgROI)
            except:
                print('No roi img generated')

            fileName_hist = os.path.join(self.read_path, f"citiusHistogram_BL{self.bl}_r{run}.npy")
            histogram = np.load(fileName_hist)
            bins = []
            counts = []
            for i in range(np.shape(histogram)[0]):
                bins.append(histogram[i][0])
                counts.append(histogram[i][1])
            bins_list.append(bins)
            counts_list.append(counts)

            try:
                fileName_histROI = os.path.join(self.read_path, f"citiusHistogram_roi_BL{self.bl}_r{run}.npy")
                histogram_roi = np.load(fileName_histROI)
                binsROI = []
                countsROI = []
                
                for i in range(np.shape(histogram_roi)[0]):
                    binsROI.append(histogram_roi[i][0])
                    countsROI.append(histogram_roi[i][1])
                binsROI_list.append(binsROI)
                countsROI_list.append(countsROI)
            except:
                print('No roi hists generated')
            
        return img_list, imgROI_list, bins_list, counts_list, binsROI_list, countsROI_list

    def find_and_fit_peaks(self, counts,
                           bin_centers = None,
                           smooth_sigma=2.0,
                           peak_prominence_frac=0.01,
                           n_peaks_needed=4,
                           plot=True):
        """
        all_values : 1D numpy array of pixel ADU values (all shots concatenated)
        nbins : histogram bins
        smooth_sigma : gaussian smoothing sigma (bins)
        peak_prominence_frac : peak prominence relative to max(hist_smoothed)
        n_peaks_needed : number of peaks to select (default: 4 => 0,1,2,3 photon)
        Returns dict with peak_positions (ADU), slope (ADU per photon), intercept, R2, fit_resids
        """
        counts = np.asarray(counts, dtype=float)
        bin_centers = np.asarray(bin_centers, dtype=float)
        if counts.shape != bin_centers.shape:
            raise ValueError("counts and bin_centers must have the same shape")
    
        # Smooth histogram (use same name hist_sm to be familiar)
        hist = counts.copy()
        hist_sm = gaussian_filter1d(hist, sigma=smooth_sigma)
    
        # 3) Find peaks
        prominence = max(1, peak_prominence_frac * np.max(hist_sm))
        peaks_idx, properties = find_peaks(hist_sm, prominence=prominence)
        peak_centers = bin_centers[peaks_idx]
        peak_heights = hist_sm[peaks_idx]
        peak_prominences = properties.get("prominences", np.zeros_like(peaks_idx))
    
        # If no peaks found, try lowering prominence a bit
        if len(peaks_idx) < n_peaks_needed:
            prominence2 = max(1, 0.5 * peak_prominence_frac * np.max(hist_sm))
            peaks_idx2, properties2 = find_peaks(hist_sm, prominence=prominence2)
            if len(peaks_idx2) > len(peaks_idx):
                peaks_idx, properties = peaks_idx2, properties2
                peak_centers = bin_centers[peaks_idx]
                peak_heights = hist_sm[peaks_idx]
                peak_prominences = properties.get("prominences", np.zeros_like(peaks_idx))
    
        if len(peaks_idx) < n_peaks_needed:
            print(f"[warning] only found {len(peaks_idx)} peaks (need {n_peaks_needed}).")
            print("Try: increasing `nbins`, lowering `peak_prominence_frac`, or narrowing ROI.")
            # still continue and attempt fit with what we have
    
        # 4) sort peaks by ADU (ascending) and pick first n_peaks_needed
        order = np.argsort(peak_centers)
        peak_centers_sorted = peak_centers[order]
        peak_heights_sorted = peak_heights[order]
        peak_prom_sorted = peak_prominences[order]
    
        selected = peak_centers_sorted[:n_peaks_needed]
        selected_heights = peak_heights_sorted[:n_peaks_needed]
        # photon numbers: 0,1,2,... for the selected peaks
        photon_numbers = np.arange(len(selected))
    
        # 5) Linear fit: ADU = slope * photon + intercept
        if len(selected) >= 2:
            coeffs = np.polyfit(photon_numbers, selected, 1)
            slope, intercept = coeffs[0], coeffs[1]
            # compute R^2
            fitted = slope * photon_numbers + intercept
            ss_res = np.sum((selected - fitted)**2)
            ss_tot = np.sum((selected - np.mean(selected))**2)
            r2 = 1 - ss_res/ss_tot if ss_tot != 0 else np.nan
        else:
            slope = intercept = r2 = np.nan
            fitted = np.array([])
    
        results = {
            "bin_centers": bin_centers,
            "hist": hist,
            "hist_smoothed": hist_sm,
            "peaks_idx": peaks_idx,
            "peak_centers_all": peak_centers,
            "selected_peak_centers": selected,
            "selected_peak_heights": selected_heights,
            "slope_adu_per_photon": slope,
            "intercept_adu": intercept,
            "r2": r2,
            "photon_numbers_used": photon_numbers,
            "fitted_values": fitted
        }

        self.slope = slope
        self.intercept = intercept
        self.calibration_calculated = True
    
        if plot:
            plt.figure(figsize=(9,5))
            plt.plot(bin_centers, hist, alpha=0.25, label="hist")
            plt.plot(bin_centers, hist_sm, lw=1.5, label=f"smoothed (σ={smooth_sigma})")
            # mark all peaks found
            plt.scatter(peak_centers, hist_sm[peaks_idx], color='C1', marker='x', label='detected peaks')
            # mark selected peaks
            plt.vlines(selected, ymin=0, ymax=np.max(hist_sm)*0.9, colors='C3', linestyles='--',
                       label=f'selected first {len(selected)} peaks')
            # annotate selected ADU values
            for i,ad in enumerate(selected):
                plt.text(ad, np.max(hist_sm)*0.92, f"{i}:{ad:.1f}", rotation=90, va='bottom', ha='center', color='C3')
            # plot fit line (ADU vs photon) as vertical ticks: show expected ADU positions for integer photons
            if not np.isnan(slope):
                photon_grid = np.arange(0, max(6, n_peaks_needed+3))
                adu_line = slope * photon_grid + intercept
                # plot ticks at these ADU positions on top of plot
                plt.vlines(adu_line, ymin=0, ymax=np.max(hist_sm)*0.6, colors='k', linestyles=':', alpha=0.6,
                           label=f'fit: {slope:.2f} ADU/photon, intercept {intercept:.1f}')
                for p,a in zip(photon_grid, adu_line):
                    plt.text(a, np.max(hist_sm)*0.62, str(p), ha='center', va='bottom', fontsize=8, color='k')
    
            plt.xlabel("ADU (intensity)")
            plt.ylabel("Counts")
            plt.title("ROI histogram — detected peaks and linear fit")
            plt.legend(loc='upper right')
            plt.yscale('log')
            plt.tight_layout()
            plt.show()
    
            print("Selected peak ADU positions (first peaks):", np.round(selected,2))
            print(f"Slope (ADU / photon): {slope:.4f}")
            print(f"Intercept (ADU): {intercept:.2f}")
            print(f"R^2 of linear fit (selected peaks): {r2:.4f}")
    
        return results

    def photon_counter(self, run_list, threshold = None, imDark = None):
        #
        print("Don't use this to count photons! the imgs read from readCITIUS .npys are divided by float(numberOfTrains)\
                 , so this is an inaccurate way of measuring photons, instead look at citiusDroplet.ipynb for droplet algorithm")
        #
        imgROI_list, binsROI_list, countsROI_list = [[] for i in range(3)]
        
        if threshold is not None:
            print("Applying threshold")
            self.threshold = threshold
        
        for run in run_list:
            # try:
            fileName_imgROI = os.path.join(self.read_path, f'citius_roi_BL{self.bl}_r{run}.npy')
            imgROI = np.load(fileName_imgROI)

            if imDark is not None:
                imgROI = imgROI - imDark

            if threshold is not None:
                #
                if threshold[-1] is None:
                    mask = (imgROI > threshold[0]) 
                #
                if threshold[-1] is not None:
                    mask = (imgROI > threshold[0]) & (imgROI < threshold[-1])
                #
                imgROI[np.where(mask == 0)] = 0.0  
                
                    
            imgROI_list.append(imgROI)
            # except:
            #     print('No roi img generated')
            #
            # try:
            fileName_histROI = os.path.join(self.read_path, f"citiusHistogram_roi_BL{self.bl}_r{run}.npy")
            histogram_roi = np.load(fileName_histROI)
            binsROI = []
            countsROI = []
            
            for i in range(np.shape(histogram_roi)[0]):
                binsROI.append(histogram_roi[i][0])
                countsROI.append(histogram_roi[i][1])
            binsROI_list.append(binsROI)
            countsROI_list.append(countsROI)
            # except:
            #     print('No roi hists generated')


        sum_imgROI = np.sum(imgROI_list, axis=0)

        int_sum_imgROI = np.sum(sum_imgROI)

        return sum_imgROI, int_sum_imgROI
            
        

        
class CITIUSProcessing():
    def __init__(self, base_path, detectorID, bl=3):

        self.bl = bl
        self.base_path = base_path
        self.detectorID = detectorID
        self.calibration_calculated = False
        self.slope = 69.0
        self.intercept = 0.0 

    def load_run_list(self, runNumbers, dark = None):
        print("I'd really recommend using readCITIUS.sh instead of this function - it is non-parallelised and very slow!")
        ### CONSTANTS
        CITIUS_IMAGE_WIDTH  = 728
        CITIUS_IMAGE_HEIGHT = 384

        ### Default values
        runNumber = -1
        lowRunNumber = -1
        highRunNumber = -1
        beamLine = 3
        # runNumbers = []
        writeableDirectory = "."
    
        upperThreshold = None
        lowerThreshold = -10
        
        for run in runNumbers:
            print(f'Loading run {run}')
            ### Get buffer
            t0 = time.time()
            buffer = ctdapy_xfel.CtrlBuffer(self.bl,run)
            sensorID = buffer.read_detidlist()[0]
            CITIUS_EMPTY_MASK = np.zeros((CITIUS_IMAGE_WIDTH,CITIUS_IMAGE_HEIGHT),dtype=np.uint8)
            buffer.read_badpixel_mask(CITIUS_EMPTY_MASK,0)
            mask   = CITIUS_EMPTY_MASK
            tagList = buffer.read_taglist()
            numberOfTrains = len(tagList)
    
            summedArray = np.zeros(np.shape(CITIUS_EMPTY_MASK))
            summedArrayNoNorm = np.zeros(np.shape(CITIUS_EMPTY_MASK))
            count,bins = np.histogram(CITIUS_EMPTY_MASK,bins=np.arange(0,1000,0.05),density=False)
            allCounts = np.zeros(np.shape(count))
    
            localSummedArray       = np.zeros(np.shape(CITIUS_EMPTY_MASK))
            localSummedArrayNoNorm = np.zeros(np.shape(CITIUS_EMPTY_MASK))
            count,bins = np.histogram(CITIUS_EMPTY_MASK,bins=np.arange(0,1000,0.05),density=False)
            localAllCounts = np.zeros(np.shape(count))
    
            nTrainsLocal = 0
            
            #### It'll be more efficient to parralise over trainIds as NTrains >> NRuns
            taglist = dbpy.read_taglist_byrun(self.bl, run)
            numIm = len(taglist)
            localTagList = taglist 
            for train in localTagList:
                I0 = getPulseEnegyInJ(self.bl,run,train)
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
                count,bins = np.histogram(citiusData,bins=np.arange(0,1000,0.05),density=False)
                localAllCounts = localAllCounts + count
                nTrainsLocal = nTrainsLocal + 1

            localSummedArray = summedArray / float(numberOfTrains)
            saveFileName = self.base_path + "/citius_BL"+ str(beamLine) + "_r" + str(run) + ".npy" 
            np.save(saveFileName,localSummedArray)

            localSummedArrayNoNorm = summedArray / float(numberOfTrains)
            saveFileName = self.base_path + "/citiusNoNormilisation_BL"+ str(beamLine) + "_r" + str(run) + ".npy" 
            np.save(saveFileName,localSummedArrayNoNorm)
    
            binCenters = []
            for i in range(np.shape(bins)[0]-1):
                binCenters.append((bins[i]+bins[i+1]) * 0.5 )
            
            saveFileName = self.base_path + "/citiusHistogram_BL"+ str(beamLine) + "_r" + str(run) + ".npy" 
            np.save(saveFileName,np.column_stack((binCenters, localAllCounts)))
            # t1 = time.time()

            # totalTime = t1-t0
            # outString = str(numberOfTrains) + " trains processed in " + str(round(totalTime,5)) + "s on " +str(nProcs) +" ranks.\n" 
        return citiusData


    def call_droplet_algorithm_ONETRAIN(self, run_list, imDark = None, threshold=None, ROI=True, out_path=None):
        # For testing, this script goes through each train in a serialised way, in future should
        # integrate into Charlie's parallelised script

        ### Default values
        beamLine = self.bl
        
        if out_path is not None:
            writeableDirectory = out_path
        else:
            writeableDirectory = '.'
        
        ### CONSTANTS
        CITIUS_IMAGE_WIDTH  = 728
        CITIUS_IMAGE_HEIGHT = 384

        x1, y1, x2, y2 = 160, 340, 230, 400  # ROI coordinates 

        imDark_roi = imDark[y1:y2, x1:x2]

        upperThreshold = threshold[-1]
        lowerThreshold = threshold[0]
        
        for run in run_list:
            ### Get buffer
            t0 = time.time()
            buffer = ctdapy_xfel.CtrlBuffer(beamLine,run)
            sensorID = buffer.read_detidlist()[0]
            
            CITIUS_EMPTY_MASK = np.zeros((CITIUS_IMAGE_WIDTH,CITIUS_IMAGE_HEIGHT),dtype=np.uint8)
            
            buffer.read_badpixel_mask(CITIUS_EMPTY_MASK,0)
            mask   = CITIUS_EMPTY_MASK
            tagList = buffer.read_taglist()
            numberOfTrains = len(tagList)
    
            summedArray = np.zeros(np.shape(CITIUS_EMPTY_MASK))
            summedArrayNoNorm = np.zeros(np.shape(CITIUS_EMPTY_MASK))
            count,bins = np.histogram(CITIUS_EMPTY_MASK,bins=np.arange(0,1000,0.05),density=False)
            localAllCounts = np.zeros(np.shape(count))
    
            localSummedArray       = np.zeros(np.shape(CITIUS_EMPTY_MASK))
                        
            localSummedArray_roi   = np.zeros(np.shape(CITIUS_EMPTY_MASK[y1:y2, x1:x2]))
                
            nTrainsLocal = 0

            results_list, citiusData_list, dropletPhotonMap_values = [], [], []
            
            # start, stop  = frames_for_rank(rank,nProcs,numberOfTrains)
            # localTagList = tagList[start:stop] 
            taglist = dbpy.read_taglist_byrun(self.bl, run)
            numIm = len(taglist)
            localTagList = taglist 
            for train in localTagList:
                if train == localTagList[0]:
                    # Applying algorithm to ROI only
                    I0 = getPulseEnegyInJ(beamLine,run,train)
                    CITIUS_EMPTY_ARRAY  = np.zeros((CITIUS_IMAGE_WIDTH,CITIUS_IMAGE_HEIGHT),dtype=np.float32)
                    buffer.read_image(CITIUS_EMPTY_ARRAY,0,train)
                    citiusData = CITIUS_EMPTY_ARRAY
                    # --------------------------------------------------------------
                    # Mask hot pixels
                    citiusData[np.where(mask == 1)] = np.nan
        
                    localSummedArray = localSummedArray + citiusData / I0 
                    
                    if ROI:                    
                        # apply roi and create seperate histogram
                        citiusData = citiusData[y1:y2, x1:x2]
                        imDark = imDark_roi
                    
                    ## Subtract dark image
                    if imDark is not None:
                        citiusData = citiusData - imDark
                    
                    ## Apply threshold 
                    if (upperThreshold != None):
                        citiusData[np.where(citiusData > upperThreshold)] = 0.0
                    
                    citiusData[np.where(citiusData < lowerThreshold)] = 0.0

                    count, bins = np.histogram(citiusData, bins=np.arange(0,1000,0.05),density=False)
                    localAllCounts = localAllCounts + count

                    # if np.amax(citiusData) > lowerThreshold:
                    # darkIm and threshold already applied in this function
                    results = self.apply_droplet_algorithm(citiusData, imDark=imDark, threshold=threshold, calibrate=False, round_photons=False)
                    
                    results_list.append(results)
                    citiusData_list.append(citiusData)
                                    
                nTrainsLocal = nTrainsLocal + 1 

            # Get an idea of speed
            t1 = time.time()
            totalTime = t1-t0
            print(f"Processing time / trains = {totalTime/len(results_list)}")

        return results_list, citiusData_list, localAllCounts, bins

    def call_droplet_algorithm(self, run_list, imDark = None, threshold=None, ROI=True, out_path=None):
        # For testing, this script goes through each train in a serialised way, in future should
        # integrate into Charlie's parallelised script

        ### Default values
        beamLine = self.bl
        
        if out_path is not None:
            writeableDirectory = out_path
        else:
            writeableDirectory = '.'
        
        ### CONSTANTS
        CITIUS_IMAGE_WIDTH  = 728
        CITIUS_IMAGE_HEIGHT = 384

        x1, y1, x2, y2 = 160, 340, 230, 400  # ROI coordinates 

        imDark_roi = imDark[y1:y2, x1:x2]

        upperThreshold = threshold[-1]
        lowerThreshold = threshold[0]
        
        for run in run_list:
            ### Get buffer
            t0 = time.time()
            buffer = ctdapy_xfel.CtrlBuffer(beamLine,run)
            sensorID = buffer.read_detidlist()[0]
            
            CITIUS_EMPTY_MASK = np.zeros((CITIUS_IMAGE_WIDTH,CITIUS_IMAGE_HEIGHT),dtype=np.uint8)
            
            buffer.read_badpixel_mask(CITIUS_EMPTY_MASK,0)
            mask   = CITIUS_EMPTY_MASK
            tagList = buffer.read_taglist()
            numberOfTrains = len(tagList)
    
            summedArray = np.zeros(np.shape(CITIUS_EMPTY_MASK))
            summedArrayNoNorm = np.zeros(np.shape(CITIUS_EMPTY_MASK))
            count,bins = np.histogram(CITIUS_EMPTY_MASK,bins=np.arange(0,1000,0.05),density=False)
            localAllCounts = np.zeros(np.shape(count))
    
            localSummedArray       = np.zeros(np.shape(CITIUS_EMPTY_MASK))
                        
            localSummedArray_roi   = np.zeros(np.shape(CITIUS_EMPTY_MASK[y1:y2, x1:x2]))
                
            nTrainsLocal = 0

            results_list, citiusData_list, dropletPhotonMap_values = [], [], []
            
            # start, stop  = frames_for_rank(rank,nProcs,numberOfTrains)
            # localTagList = tagList[start:stop] 
            taglist = dbpy.read_taglist_byrun(self.bl, run)
            numIm = len(taglist)
            localTagList = taglist 
            for train in localTagList:
                # if train < localTagList[10]:
                # Applying algorithm to ROI only
                I0 = getPulseEnegyInJ(beamLine,run,train)
                CITIUS_EMPTY_ARRAY  = np.zeros((CITIUS_IMAGE_WIDTH,CITIUS_IMAGE_HEIGHT),dtype=np.float32)
                buffer.read_image(CITIUS_EMPTY_ARRAY,0,train)
                citiusData = CITIUS_EMPTY_ARRAY
                # --------------------------------------------------------------
                # Mask hot pixels
                citiusData[np.where(mask == 1)] = np.nan
    
                localSummedArray = localSummedArray + citiusData / I0 
                
                if ROI:                    
                    # apply roi and create seperate histogram
                    citiusData = citiusData[y1:y2, x1:x2]
                    imDark = imDark_roi
                
                ## Subtract dark image
                if imDark is not None:
                    citiusData = citiusData - imDark
                
                ## Apply threshold 
                if (upperThreshold != None):
                    citiusData[np.where(citiusData > upperThreshold)] = 0.0
                
                citiusData[np.where(citiusData < lowerThreshold)] = 0.0

                count, bins = np.histogram(citiusData, bins=np.arange(0,1000,0.05),density=False)
                
                localAllCounts = localAllCounts + count

                # if np.amax(citiusData) > lowerThreshold:
                # darkIm and threshold already applied in this function
                results = self.apply_droplet_algorithm(citiusData, imDark=imDark, threshold=threshold, calibrate=False, round_photons=False)
                
                results_list.append(results)
                citiusData_list.append(citiusData)
                                    
                nTrainsLocal = nTrainsLocal + 1 

            # Get an idea of speed
            t1 = time.time()
            totalTime = t1-t0
            print(f"Processing time / train = {totalTime/len(results_list)}")

        return results_list, citiusData_list, localAllCounts, bins

    def apply_droplet_algorithm(self, 
                                img, 
                                imDark=None,
                                threshold=None,
                                calibrate=False,
                                min_size=1,
                                connectivity=2,
                                round_photons=False,
                                histogram_bins=np.arange(0,1000,0.05)):
        """
        Process files an img using droplet algorithm.
        
        Returns
        -------
        results : dict
            Keys:
              'droplets' : dict -> list of droplet dicts (keys: 'y','x','n_pix','sum_adu','photons')
              'photon_maps' : dict -> 2D photon map (float)
              'histograms' : dict -> (bin_centers, counts)
        """
        results = {
            'droplets': {},
            'photon_maps': {},
            'histograms': {}
        }
    
        # helper labeling structure
        structure = ndimage.generate_binary_structure(2, connectivity)

        # plt.imshow(img)
        # plt.show()

        mask = np.zeros((np.shape(img)), dtype=np.uint8)
        # mask = np.zeros((np.shape(img)))

        # optional dark subtraction
        if imDark is not None:
            # print("Dark image subtracted")
            if imDark.shape != img.shape:
                raise ValueError("darkIm shape mismatch with image")
            img = img - imDark

        if threshold is not None:
            # print(f"Applying threshold: {threshold}")
            #
            if threshold[-1] is None:
                mask = (img > threshold[0]) 
            #
            if threshold[-1] is not None:
                mask = (img > threshold[0]) & (imgROI < threshold[-1])
                
        img[np.where(mask == 0)] = 0.0  

        droplets = []
        
        photon_map = np.zeros((np.shape(img)), dtype=np.uint8)
        photon_totals = []

        structure = ndimage.generate_binary_structure(2, connectivity)  # 4- or 8-neigh
        labeled, nlabels = ndimage.label(mask, structure=structure)
    
        if nlabels == 0:
            # fill results with empty/zero but consistent types
            results['droplets'] = []
            results['photon_maps'] = photon_map
            
            counts = np.zeros(len(histogram_bins)-1, dtype=int)
            bin_centers = 0.5 * (histogram_bins[:-1] + histogram_bins[1:])
            results['histograms'] = [bin_centers, counts]
            
            return results
    
        # region sizes:
        sizes = np.bincount(labeled.ravel())
        # sizes[0] is background count
        labels = np.arange(1, nlabels+1)

        if nlabels > 0:
            # bincount approach for sizes and quick skip of background
            sizes = np.bincount(labeled.ravel())
            for lab in range(1, nlabels + 1):
                n_pix = int(sizes[lab]) if lab < len(sizes) else 0
                if n_pix < min_size:
                    continue
                ys, xs = np.nonzero(labeled == lab)
                vals = img[ys, xs]
                sum_adu = float(vals.sum())
                max_adu = float(vals.max())
                
                # centroid (intensity-weighted)
                if sum_adu != 0:
                    cy = float((ys * vals).sum() / sum_adu)
                    cx = float((xs * vals).sum() / sum_adu)
                else:
                    cy, cx = float(ys.mean()), float(xs.mean())

                if calibrate:
                    if not hasattr(self, 'slope') or not hasattr(self, 'intercept'):
                        raise AttributeError("self.slope and self.intercept must be set before calling droplet algorithm")
                    # convert sum ADU -> photons using calibration
                    photons = sum_adu / float(self.slope)
                else:
                    photons = sum_adu
                
                if round_photons:
                    photons_int = int(np.rint(max(0.0, photons)))
                    photons_use = float(photons_int)
                else:
                    photons_use = float(max(0.0, photons))

                # assign to photon_map at nearest integer pixel of centroid
                iy = int(round(cy)); ix = int(round(cx))
                if 0 <= iy < img.shape[0] and 0 <= ix < img.shape[1]:
                    photon_map[iy, ix] += photons_use

                droplets.append({
                    'y': cy, 'x': cx,
                    'n_pix': n_pix,
                    'sum_adu': sum_adu,
                    'max_adu': max_adu,
                    'photons': photons_use
                })
                photon_totals.append(photons_use)

        # histogram of droplet photons (use returned photon_totals)(safe if empty)
        if len(photon_totals) == 0:
            counts = np.zeros(len(histogram_bins)-1, dtype=int)
        else:
            counts, edges = np.histogram(photon_totals, bins=histogram_bins)
        bin_centers = 0.5 * (histogram_bins[:-1] + histogram_bins[1:])

        # store results
        results['droplets'] = droplets
        
        # print(droplets)
        
        results['photon_maps'] = photon_map
        
        results['histograms'] = [bin_centers, counts]

        # print(f"why am I not finding {results}")
        
        return results

    

        