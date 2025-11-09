### Eta calculation
import dbpy 
from citius import *
from mpccd import *
import numpy as np
import matplotlib.pyplot as plt
def plot_roi(im, center_x, center_y, roi_size):
    # Calculate top-left corner of the ROI
    half = roi_size // 2
    top_left_x = center_x - half
    top_left_y = center_y - half

    plt.figure()
    plt.imshow(im)
    plt.colorbar()
    # Draw a red rectangle around ROI
    rect = plt.Rectangle((top_left_x, top_left_y), roi_size, roi_size,
                     linewidth=1, edgecolor='red', facecolor='none')

    plt.gca().add_patch(rect)

    # Draw horizontal and vertical center lines
    plt.axhline(y=center_y, color='red', linestyle='--', linewidth=0.5)
    plt.axvline(x=center_x, color='red', linestyle='--', linewidth=0.5)

    im_roi = im[center_y - half:center_y + half, center_x - half:center_x + half]
    print('summed intensity in ROI', np.sum(im_roi))
    plt.figure()
    plt.imshow(im_roi)
    plt.colorbar()

def get_ROI(im_all, center_x, center_y, roi_size):
    half = roi_size // 2
    top_left_x = center_x - half
    top_left_y = center_y - half
   
    im_roi = im_all[center_y - half:center_y + half, center_x - half:center_x + half]
    im_roi_avg = np.nanmean(im_roi)
    im_roi_sum = np.nansum(im_roi)
    return im_roi, im_roi_avg, im_roi_sum

def get_equip_data_perrun(equip_ID, run):
    beamline = 3
    taglist = dbpy.read_taglist_byrun(beamline, run)
    highTag = dbpy.read_hightagnumber(beamline, run)
    motor_pos = np.array(dbpy.read_syncdatalist_float(equip_ID, highTag,taglist))
    return {'equip_ID':equip_ID, 'motor_pulse': motor_pos, 'tag_list': self.taglist, 'run': self.run}

def getIncomingFlux(run):
    
    bm_calib = get_equip_data_perrun('xfel_bl_3_st_4_bm_1_pd/charge_calib_in_joule', 1)
    beamIN = get_equip_data_perrun('xfel_bl_3_st_4_bm_1_pd/charge', 1)

    energyBeforeC1 = beamIn / bm_calib
    numberOfPhotonsBeforeC1 = energyBeforeC1 / (1.6e-19 * 1e4)
    return numberOfPhotonsBeforeC1

## Try new calibration for beam monitor using MPCCD 1's data
### Do it pulse by pulse
beamline = 3
run = 1616892
taglist = dbpy.read_taglist_byrun(beamline, run)
highTag = dbpy.read_hightagnumber(beamline, run)
import matplotlib.patches as patches
roi = [195,370,30]
CITIUS_IMAGE_WIDTH  = 728
CITIUS_IMAGE_HEIGHT = 384
print(f'Loading run {run}')
### Get buffer
t0 = time.time()
buffer = ctdapy_xfel.CtrlBuffer(3,run)
sensorID = buffer.read_detidlist()[0]
CITIUS_EMPTY_MASK = np.zeros((CITIUS_IMAGE_WIDTH,CITIUS_IMAGE_HEIGHT),dtype=np.uint8)
buffer.read_badpixel_mask(CITIUS_EMPTY_MASK,0)
mask   = CITIUS_EMPTY_MASK
tagList = buffer.read_taglist()
numberOfTrains = len(tagList)

upperThreshold = None
lowerThreshold = 5

upperDiode = dbpy.read_syncdatalist_float('xfel_bl_3_st_4_bm_1_pd_upper_fitting_peak/voltage', highTag,taglist)
i = 0
ratio = []

shutter = dbpy.read_syncdatalist_float('xfel_bl_3_shutter_1_open_valid/status', highTag, taglist)
for tag in tagList:
    train = tag
    if (shutter[i] == 0):
        continue
    CITIUS_EMPTY_ARRAY  = np.zeros((CITIUS_IMAGE_WIDTH,CITIUS_IMAGE_HEIGHT),dtype=np.float32)
    buffer.read_image(CITIUS_EMPTY_ARRAY,0,train)
    citiusData = CITIUS_EMPTY_ARRAY
    
    ### Apply threshold 
    if (upperThreshold != None):
        citiusData[np.where(citiusData > upperThreshold)] = np.nan
    citiusData[np.where(citiusData < lowerThreshold)] = 0.0
        ### Mask hot pixels
    citiusData[np.where(mask == 1)] = np.nan   

    voltageOnDiode   =  upperDiode[i]
    
    intensityInROI   = np.nansum(citiusData) / 69.2 ## in number of photons
    intensityOnDiode = 0.00864439 * voltageOnDiode - 0.0086778 ## in micro Joule

    numberOfPhotonsComingIn = intensityOnDiode * 1e6 / (1.6e-19 * 1e4)
    
    ratio.append(intensityInROI / numberOfPhotonsComingIn)
    
    i = i + 1

