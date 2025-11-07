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

class CITIUSProcessing():
    def __init__(self, base_path, detectorID, bl=3):
        """
        Initialize calibration class.
        
        Parameters
        ----------
        base_path : str
            Path to the main analysis folder containing subdirs for spectrometers.
        detector_name : str
            Identifier for spectrometer (used in subfolder names).
        """
        self.bl = bl
        self.base_path = base_path
        self.detectorID = detectorID
        self.calibration_calculated = False
        self.slope = 1.0
        self.intercept = 0.0 

    def load_run_list(self, runNumbers, dark = None):
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
            count,bins = np.histogram(CITIUS_EMPTY_MASK,bins=np.arange(0,100,0.05),density=False)
            allCounts = np.zeros(np.shape(count))
    
            localSummedArray       = np.zeros(np.shape(CITIUS_EMPTY_MASK))
            localSummedArrayNoNorm = np.zeros(np.shape(CITIUS_EMPTY_MASK))
            count,bins = np.histogram(CITIUS_EMPTY_MASK,bins=np.arange(0,100,0.05),density=False)
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
                count,bins = np.histogram(citiusData,bins=np.arange(0,100,0.05),density=False)
                localAllCounts = localAllCounts + count
                nTrainsLocal = nTrainsLocal + 1

            summedArray = summedArray / float(numberOfTrains)
            saveFileName = self.base_path + "/citius_BL"+ str(beamLine) + "_r" + str(run) + ".npy" 
            np.save(saveFileName,summedArray)
    
            saveFileName = self.base_path + "/citiusNoNormilisation_BL"+ str(beamLine) + "_r" + str(run) + ".npy" 
            np.save(saveFileName,summedArrayNoNorm)
    
            binCenters = []
            for i in range(np.shape(bins)[0]-1):
                binCenters.append((bins[i]+bins[i+1]) * 0.5 )
            
            saveFileName = self.base_path + "/citiusHistogram_BL"+ str(beamLine) + "_r" + str(run) + ".npy" 
            np.save(saveFileName,np.column_stack((binCenters, localAllCounts)))
            # t1 = time.time()

            # totalTime = t1-t0
            # outString = str(numberOfTrains) + " trains processed in " + str(round(totalTime,5)) + "s on " +str(nProcs) +" ranks.\n" 
        return citiusData

    def find_droplets(frame, threshold, slope=1.0, intercept=0.0,
                  dark_subtracted=True, min_size=1, connectivity=2):
        """
        Basic droplet finder for a single 2D frame (dark/pedestal corrected).
        Returns list of (y, x, photons).
        """
        img = frame.copy().astype(float)
        mask = img > threshold
        structure = ndimage.generate_binary_structure(2, connectivity)
        labeled, nlabels = ndimage.label(mask, structure)
        if nlabels == 0:
            return []

        droplets = []
        for label in range(1, nlabels + 1):
            ys, xs = np.nonzero(labeled == label)
            vals = img[ys, xs]
            n_pix = len(vals)
            if n_pix < min_size:
                continue
            sum_adu = float(np.sum(vals))
            photons = (sum_adu - intercept * n_pix) / slope if not dark_subtracted else sum_adu / slope
            photons = max(0.0, photons)
            cy = np.sum(ys * vals) / sum_adu
            cx = np.sum(xs * vals) / sum_adu
            droplets.append((cy, cx, photons))
        return droplets

    from scipy import ndimage

    def apply_droplet_algorithm(self, img, run, darkIm=None,
                            threshold_adus=None,
                            n_sigma_threshold=0.0,
                            min_size=1,
                            connectivity=2,
                            round_photons=True,
                            save_photon_maps=False,
                            out_dir=None,
                            histogram_bins=np.linspace(0,6,61)):
        """
        Process files for runs in run_list using droplet algorithm.
        Assumes each run corresponds to a single .npy image file at:
            self.base_path + f'/citius_BL{self.bl}_r{run}.npy'
        Parameters
        ----------
        run_list : list of ints
            Runs to process.
        darkIm : 2D numpy array or None
            Dark/pedestal image to subtract (same shape as images).
        min_size : int
            Minimum number of connected pixels to keep a droplet.
        connectivity : 1 or 2
            1 -> 4-neighbour, 2 -> 8-neighbour connectivity.
        round_photons : bool
            Round photon estimate to nearest integer before assigning to map.
        save_photon_maps : bool
            If True, save per-run photon_map as .npy to out_dir (must be provided).
        out_dir : str or None
            Directory to save outputs if save_photon_maps True.
        histogram_bins : 1d array
            Bins for droplet-photon histogram.
        Returns
        -------
        results : dict
            Keys:
              'droplets' : dict run -> list of droplet dicts (keys: 'y','x','n_pix','sum_adu','photons')
              'photon_maps' : dict run -> 2D photon map (float)
              'histograms' : dict run -> (bin_centers, counts)
        """
        results = {
            'droplets': {},
            'photon_maps': {},
            'histograms': {}
        }
    
        # helper labeling structure
        structure = ndimage.generate_binary_structure(2, connectivity)
    
        # for run in run_list:
            # fname = self.base_path + f'/citius_BL{self.bl}_r{run}.npy'
            # try:
            #     img = np.load(fname).astype(float)
            # except Exception as e:
            #     print(f"[WARN] could not load {fname}: {e}")
            #     continue

        # plt.imshow(img)
        # plt.show()

        # optional dark subtraction
        if darkIm is not None:
            if darkIm.shape != img.shape:
                raise ValueError("darkIm shape mismatch with image")
            img = img - darkIm

        # compute robust threshold if none provided
        if threshold_adus is None:
            # robust location + scale: median & MAD -> sigma ~ 1.4826*MAD
            med = np.nanmedian(img)
            mad = np.nanmedian(np.abs(img - med))
            sigma_est = 1.4826 * (mad if mad > 0 else np.nanstd(img))
            thr = med + n_sigma_threshold * sigma_est
        else:
            thr = float(threshold_adus)


        # boolean mask of candidate pixels
        mask = img > thr

        # plt.imshow(mask)
        # plt.colorbar()
        # plt.show()

        # label connected components
        labeled, nlabels = ndimage.label(mask, structure=structure)

        droplets = []
        # prepare photon map (same shape)
        photon_map = np.zeros_like(img, dtype=float)
        photon_totals = []

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

                # convert sum ADU -> photons using calibration
                # Here we assume img already had dark/pedestal removed (we subtracted darkIm above).
                # So use photons = sum_adu / slope
                if not hasattr(self, 'slope') or not hasattr(self, 'intercept'):
                    raise AttributeError("self.slope and self.intercept must be set before calling droplet algorithm")
                photons = sum_adu / float(self.slope)
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

        # histogram of droplet photons (use returned photon_totals)
        counts, edges = np.histogram(photon_totals, bins=histogram_bins)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

        # store results
        results['droplets'][run] = droplets
        results['photon_maps'][run] = photon_map
        results['histograms'][run] = (bin_centers, counts)

        # optional save
        if save_photon_maps:
            if out_dir is None:
                raise ValueError("out_dir must be provided when save_photon_maps=True")
            outname = f"{out_dir}/citius_photon_map_BL{self.bl}_r{run}.npy"
            np.save(outname, photon_map)
    
        return results

    

        