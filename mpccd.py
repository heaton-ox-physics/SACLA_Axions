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

class MPCCDProcessing():
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
        self.slope = None
        self.intercept = None

    def read_det(self, run, calibrate):
        taglist = dbpy.read_taglist_byrun(self.bl, run)
        numIm = len(taglist)
        print('Run: {}\nNumber of images: {}\nDetector ID: {}'.format(run, numIm, self.detectorID))
        
        #stpy.StorageReader(detectorID, bl, run_numbers)
        #run_numbers: tuple of run list
        obj = stpy.StorageReader(self.detectorID, self.bl, (run,))
        buff = stpy.StorageBuffer(obj)
        obj.collect(buff, taglist[0])
        im2D = buff.read_det_data(0)
    
        im2Dall = np.zeros((numIm, len(im2D[:,0]), len(im2D[0,:])))
        im2Dall[0] = im2D.copy()
    
        i = 1
        for tag in taglist[1:]:
            if i % 100 == 0:
                sys.stdout.write('\r%d' % i)
                sys.stdout.flush()
            obj.collect(buff, tag)
            im2Dall[i] = buff.read_det_data(0).copy()
            
            if calibrate:
                im2Dall[i] = (im2Dall[i] - self.intercept) / self.slope
                
            i += 1
        
        return im2Dall

    def read_list_runs(self, runlist, calibrate=False):
        im2Dall_list = []
        for run in runlist:
            taglist = dbpy.read_taglist_byrun(self.bl, run)
            numIm = len(taglist)
            print('Run: {}\nNumber of images: {}\nDetector ID: {}'.format(run, numIm, self.detectorID))
            
            # obj = stpy.StorageReader(detectorID, bl, run_numbers)
            #run_numbers: tuple of run list
            print(run)
            obj = stpy.StorageReader(self.detectorID, self.bl, (run,))
            buff = stpy.StorageBuffer(obj)
            obj.collect(buff, taglist[0])
            im2D = buff.read_det_data(0)
        
            im2Dall = np.zeros((numIm, len(im2D[:,0]), len(im2D[0,:])))
            im2Dall[0] = im2D.copy()
        
            i = 1
            for tag in taglist[1:]:
                if i % 100 == 0:
                    sys.stdout.write('\r%d' % i)
                    sys.stdout.flush()
                obj.collect(buff, tag)
                im2Dall[i] = buff.read_det_data(0).copy()
                
                if calibrate:
                    im2Dall[i] = (im2Dall[i] - self.intercept) / self.slope
                    
                i += 1
            
        
            im2Dall_list.append(im2Dall)
        return im2Dall_list

    def read_det_sbt(self, run, imDark=None, threshold=None, calibrate=False):
        taglist = dbpy.read_taglist_byrun(self.bl, run)
        numIm = len(taglist)
        print('\nRun: {}\nNumber of images: {}\nDetector ID: {}'.format(run, numIm, self.detectorID))
        
        #stpy.StorageReader(detectorID, bl, run_numbers)
        #run_numbers: tuple of run list
        obj = stpy.StorageReader(self.detectorID, self.bl, (run,))
        buff = stpy.StorageBuffer(obj)
        obj.collect(buff, taglist[0])
        im2D = buff.read_det_data(0)
    
        im2Dall_sbt = np.zeros((numIm, len(im2D[:,0]), len(im2D[0,:])))

        if imDark is not None:
            im2Dall_sbt[0] = im2D - imDark
    
        i = 1
        for tag in taglist[1:]:
            if i % 100 == 0:
                sys.stdout.write('\r%d' % i)
                sys.stdout.flush()
            obj.collect(buff, tag)
            
            if imDark is not None:
                im2Dall_sbt[i] = buff.read_det_data(0) - imDark
            
            if calibrate:
                im2Dall_sbt[i] = (im2Dall_sbt[i] - self.intercept) / self.slope

            if threshold is not None:
                if threshold[-1] is not None: 
                    # # print(f'threshold: {threshold[0]}')
                    mask = (im2Dall_sbt[i] > threshold[0]) & (im2Dall_sbt[i] < threshold[-1])
                if threshold[-1] is None:
                    # print(f'threshold: {threshold[0]}')
                    mask = (im2Dall_sbt[i] > threshold[0])
                im2Dall_sbt[i][np.where(mask == 0)] = np.nan
            i += 1
        
        return im2Dall_sbt

    # THIS FUNCTION IS DEPRECATED - use create_average_dark then load_multiple_images instead
    def load_images(self, run = None, runDark = None, threshold = None, calibrate=False):
        #
        if runDark is not None:
            print(f'Creating dark run image')
            imDark = np.mean(self.read_det(runDark, calibrate = calibrate),0)
            self.imDark = imDark
            im2Dall = self.read_det_sbt(run, imDark=imDark, threshold = threshold, calibrate = calibrate)
        else:
            im2Dall = self.read_det(run, calibrate = calibrate)

        im2Dave = np.mean(im2Dall, 0)
        
        return im2Dall, im2Dave

    def create_average_dark(self, run_list, threshold = None, calibrate=False):
        im2Dave_list, im2Dall_list = [], []
        for run in run_list:
            
            im2Dall = self.read_det_sbt(run, imDark=None, threshold = threshold, calibrate = calibrate)
            im2Dave = np.mean(im2Dall, 0)
            
            im2Dall_list.append(im2Dall)
            im2Dave_list.append(im2Dave)
            
        return im2Dall_list, im2Dave_list

    def load_multiple_images(self, run_list, imDark = None, threshold = None, calibrate=False):
        im2Dave_list, im2Dall_list = [], []
        for run in run_list:
            if imDark is not None:
                print(f'Applying imDark')
                im2Dall = self.read_det_sbt(run, imDark, threshold = threshold, calibrate = calibrate)
            else:
                im2Dall = self.read_det_sbt(run, imDark = None, threshold = threshold, calibrate = calibrate)
    
            im2Dave = np.mean(im2Dall, 0)

            # plt.figure()
            # plt.imshow(im2Dave)
            # plt.show()
            
            im2Dall_list.append(im2Dall)
            im2Dave_list.append(im2Dave)
        
        return im2Dall_list, im2Dave_list

    def create_run_histograms(self, x1, y1, x2, y2, run_list, imDark=None, bins=np.arange(0,100,0.05)):
        all_values_list = []
        total_counts = np.zeros(len(bins) - 1, dtype=float)  # combined histogram counts
        for run in run_list:
            #
            # imDark = np.mean(self.read_det(runDark, calibrate=False),0)
            #
            taglist = dbpy.read_taglist_byrun(self.bl, run)
            numIm = len(taglist)
            print('\nRun: {}\nNumber of images: {}\nDetector ID: {}'.format(run, numIm, self.detectorID))
            
            #stpy.StorageReader(detectorID, bl, run_numbers)
            #run_numbers: tuple of run list
            obj = stpy.StorageReader(self.detectorID, self.bl, (run,))
            buff = stpy.StorageBuffer(obj)
            obj.collect(buff, taglist[0])
            im2D = buff.read_det_data(0)
        
            im2Dall_sbt = np.zeros((numIm, len(im2D[:,0]), len(im2D[0,:])))
            print(np.shape(im2Dall_sbt))
            im2Dall_roi = np.zeros((numIm, (y2-y1), (x2-x1)))
            
            all_values = []
            hist_counts = []
    
            print(f'Dark runs subtracted')
            im2Dall_sbt[0] = im2D - imDark
    
            run_counts = np.zeros_like(total_counts)
        
            i = 1
            for tag in taglist[1:]:
                if i % 100 == 0:
                    sys.stdout.write('\r%d' % i)
                    sys.stdout.flush()
                obj.collect(buff, tag)
                im2Dall_sbt[i] = buff.read_det_data(0) - imDark
                im2Dall_roi[i] = im2Dall_sbt[i,y1:y2, x1:x2]
                
                all_values.append(im2Dall_roi[i].ravel())
                
                counts, bins = np.histogram(im2Dall_roi, bins=bins, density=False)
                hist_counts.append(counts)
    
                run_counts += counts
    
                i += 1
            all_values = np.concatenate(all_values)
            
            total_counts += run_counts
            all_values_list.append(all_values)
            
        return all_values_list, total_counts



    def find_and_fit_peaks(self, all_values,
                           nbins=1000,
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
        # 1) Histogram
        vals = all_values
        hist, bin_edges = np.histogram(vals, bins=nbins, density=False)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
        # 2) Smooth histogram to help peak finding
        hist_sm = gaussian_filter1d(hist.astype(float), sigma=smooth_sigma)
    
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
            plt.yscale('linear')
            plt.tight_layout()
            plt.show()
    
            print("Selected peak ADU positions (first peaks):", np.round(selected,2))
            print(f"Slope (ADU / photon): {slope:.4f}")
            print(f"Intercept (ADU): {intercept:.2f}")
            print(f"R^2 of linear fit (selected peaks): {r2:.4f}")
    
        return results

    def load_images_with_calibration(self, run = None, runDark = None):
        #
        if not self.calibration_calculated:
            print("You haven't generate your calibration dummy! Run func find_and_fit_peaks")
        #
        if runDark is not None:
            print(f'Creating dark run image')
            imDark = np.mean(self.read_det(runDark, calibrate=True),0)
            im2Dall = self.read_det_sbt(run, imDark, calibrate=True)
        else:
            im2Dall = self.read_det(run, calibrate=True)

        print(f'Averaging (run - dark_run)')
        im2Dave = np.mean(im2Dall, 0)        
        
        return im2Dall, im2Dave
        
        
