#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SBO Oximetry Analysis GUI
=========================

A graphical user interface for analyzing susceptibility-based oximetry (SBO) 
MRI data to measure blood oxygen saturation (HbO2) in cerebral vessels.

Description
-----------
This application provides tools for:
    - Loading and displaying dual-echo gradient-echo MRI data from Philips 
      PAR/REC or NIfTI formats
    - Manual and automatic ROI delineation for vessel (sinus) and tissue 
      segmentation
    - Blood oxygen saturation (HbO2) quantification from inter-echo phase 
      difference
    - Selective phase unwrapping to correct wrapping artefacts
    - Vessel-to-B0 angle estimation from angiographic MRI data
    - Data export to CSV

Susceptibility-Based Oximetry Method
-------------------------------------
The SBO method measures blood oxygen saturation by exploiting the paramagnetic 
properties of deoxyhaemoglobin, which induces a susceptibility difference 
between blood and surrounding tissue. HbO2 is calculated from the inter-echo 
phase difference in a vessel ROI and a tissue ROI:

    HbO2 = 1 - 2*(Δφ_vessel - Δφ_tissue) / 
            (γ * 4π * Δχ_do * B0 * (cos²θ - 1/3) * Hct * ΔTE) - Δχ_0

Reference: Jain et al. Journal of Cerebral Blood Flow & Metabolism 2010, 
Vol.30(9) 1598-1607, https://doi.org/10.1038/jcbfm.2010.49

Supported Input Formats
-----------------------
    - Philips PAR/REC files
    - NIfTI files (.nii): Converted from DICOM using dcm2niix 
      (https://github.com/rordenlab/dcm2niix). Requires accompanying JSON sidecar.

System Requirements
-------------------
    - Python 3.8+
    - Dependencies: tkinter, matplotlib, nibabel, numpy, pandas, scipy, 
      scikit-image, PIL, mpl_point_clicker

Tested Scanners
---------------
    - 3T Philips dSTREAM Achieva MRI
    - 3T Siemens Biograph mMR hybrid PET/MR system

Usage
-----
Command line:
    $ python SBOCalculator.py

With pre-specified files:
    $ python SBOCalculator.py --img <path_to_PAR_file>
    $ python SBOCalculator.py --img <path_to_PAR_file> --hct 0.43 --angle 10.5
    $ python SBOCalculator.py --img <path_to_PAR_file> --angio <path_to_angio_PAR>

Keyboard Shortcuts
------------------
    Ctrl+1      : Display Echo 1 image
    Ctrl+2      : Display Echo 2 image
    Ctrl+3      : Display Echo difference (Echo 2 - Echo 1)
    Ctrl+4      : Display Modulus image
    Ctrl+5      : Display ROI mask
    Ctrl+Q/W/E  : Change colormap (jet, gray, viridis)
    Ctrl+N      : Load new PAR/REC file
    Ctrl+R      : Load saved ROI file
    Ctrl+S      : Save data
    Ctrl+O      : Open vessel angle estimation module
    Ctrl+A      : About
    Up/Down     : Navigate through frames

Output Files
------------
    - CSV file: HbO2, Sinus phase, Tissue phase, Hematocrit, Vessel angle 
      per frame
    - NPZ file (optional): ROI polygon coordinates and binary masks for 
      Sinus and Tissue ROIs
    - NIfTI file (optional): Labelled ROI mask in original image space
      (Sinus=1, Tissue=2)
    - GIF file (optional): Animation of ROI overlays across frames

References
----------
    Jain V et al. JCBFM 2010;30(9):1598-1607
    Vestergaard et al. JCBFM 2019;39(5):834-848
    Vestergaard et al. Cerebral Cortex 2022;32(6):1295-1306

Author
------
    Mark B. Vestergaard
    Functional Imaging Unit
    Department of Clinical Physiology and Nuclear Medicine
    Rigshospitalet Copenhagen, Denmark
    mark.bitsch.vestergaard@regionh.dk

"""

# =============================================================================
# IMPORTS
# =============================================================================
from tkinter import (Tk, filedialog, END, Menu, LabelFrame, Label, Entry, 
                     Button, Scale, Frame, Grid, StringVar, Checkbutton)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import nibabel as nib
import argparse
import numpy as np
import numpy.matlib
import scipy
import scipy.ndimage
import pandas as pd
from matplotlib.widgets import PolygonSelector
from matplotlib.patches import Polygon 
from matplotlib.path import Path
from skimage import measure
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw
from mpl_point_clicker import clicker
import os 
import math
import glob


# =============================================================================
# COMMAND LINE ARGUMENT PARSING
# =============================================================================
# Command line arguments for batch processing or scripted usage
parser = argparse.ArgumentParser(
    description='Calculate blood oxygen saturation (HbO2) from susceptibility-based oximetry MRI data',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='''
Examples:
  %(prog)s                                    # Interactive file selection
  %(prog)s --img scan.PAR                     # Load Philips PAR/REC file
  %(prog)s --img scan.PAR --hct 0.43          # Specify hematocrit
  %(prog)s --img scan.PAR --angle 10.5        # Specify vessel angle (degrees)
  %(prog)s --img scan.PAR --angio angio.PAR   # Load angiographic data for angle estimation
    '''
)
parser.add_argument('--img', dest='img_input', 
                    help='Input SBO image (PAR/REC format)')
parser.add_argument('--angio', dest='angio_input', 
                    help='Input angiographic image for vessel angle estimation (PAR/REC format)')
parser.add_argument('--hct', dest='hct_input', 
                    help='Hematocrit value (default: 0.43)')
parser.add_argument('--angle', dest='ss_angle_input', 
                    help='Vessel angle relative to B0 in degrees (default: 0)')
args = parser.parse_args()


# =============================================================================
# GUI WINDOW INITIALIZATION
# =============================================================================
# Create the main tkinter GUI window with responsive sizing
window = Tk()
window.title('SBOCalculator')
GUI_width = window.winfo_screenwidth() * 0.8
GUI_height = window.winfo_screenheight() * 1
window.geometry(str(int(GUI_width)) + 'x' + str(int(GUI_height)))
window.resizable(True, True)


# =============================================================================
# FILE LOADING
# =============================================================================
# Flag to track if working with NIfTI files (affects processing pipeline)
nifti_files = None

# Load SBO data - either from command line argument or via file dialog
if args.img_input is None:
    # No command line arguments - open file dialog
    print('Select SBO file')
    raw_img_filename = filedialog.askopenfilename(
        initialdir="/", title="Select file",
        filetypes=(("Philips PAR", "*.PAR"), ("NIFTI", "*.nii"), ("all files", "*.*")),
        multiple=True
    )        
    args.img_input = raw_img_filename
    file_type = os.path.splitext(raw_img_filename[0])[1]
    if file_type == '.nii':
        nifti_files = True
        import json
else:
    # PAR/REC file provided via command line
    raw_img_filename = [args.img_input]

    
# =============================================================================
# IMAGE DATA CLASSES
# =============================================================================

class Img_data_parrec():
    """
    Class to handle Philips PAR/REC format SBO image data.
    
    Loads and processes dual-echo gradient-echo MRI data from Philips scanners,
    extracting phase images at each echo time and the modulus image. If the 
    acquisition includes velocity-encoded frames (combined SBO + PCM), these 
    are automatically identified and removed.
    
    Attributes
    ----------
    raw_img : nibabel.parrec.PARRECImage
        Raw loaded image object from nibabel
    raw_img_filename : str
        Path to the source PAR file
    img : nibabel image
        Canonical orientation image
    nifti_file : nibabel.Nifti1Image
        NIfTI representation for compatibility
    pold_header : dict
        General PAR header information (offcenter, angulation, FOV, etc.)
    pold_header_slice_info : dict
        Per-slice PAR header information (image types, echo times, etc.)
    Venc : float
        Velocity encoding value (cm/s) - extracted from header
    Echo1_deck : ndarray
        3D phase image array from echo 1 (x, y, frames)
    Echo2_deck : ndarray
        3D phase image array from echo 2 (x, y, frames)
    Echo_diff_deck : ndarray
        3D inter-echo phase difference (Echo2 - Echo1)
    Mod_deck : ndarray
        3D modulus image array, scaled to [-π, π] range
    DeltaTE : float
        Echo time difference in seconds
    ndix : tuple
        Shape of the echo image arrays (rows, cols, n_frames)
    
    Methods
    -------
    set_new_data()
        Load a new PAR/REC file interactively
    """
    def __init__(self, raw_img_filename):
        # Parse PAR header for slice-level metadata
        self.hdr_obj = open(raw_img_filename) 
        self.pold_header, self.pold_header_slice_info = nib.parrec.parse_PAR_header(self.hdr_obj)
        
        # Load image data using nibabel
        self.raw_img = nib.load(raw_img_filename)
        self.raw_img_filename = raw_img_filename
        self.img = nib.as_closest_canonical(self.raw_img)
        self.nifti_file = nib.Nifti1Image(self.img.dataobj, self.img.affine, header=self.img.header)
        self.Venc = self.img.header.general_info.get('phase_enc_velocity')[2]
    
        # Compute phase images from real and imaginary components
        self.Phase_deck_tmp = np.rot90(np.arctan2(
            self.img.dataobj[:, :, 0, self.pold_header_slice_info['image_type_mr'] == 2],
            self.img.dataobj[:, :, 0, self.pold_header_slice_info['image_type_mr'] == 1]
        ))
        
        # Extract and scale modulus image to [-π, π] range
        self.Mod_deck_tmp = np.rot90(self.img.dataobj[:, :, 0, np.logical_and(
            self.pold_header_slice_info['image_type_mr'] == 0, 
            self.pold_header_slice_info['echo number'] == 1
        )])
        self.Mod_deck = ((self.Mod_deck_tmp / self.Mod_deck_tmp.max()) * 2 * math.pi) - math.pi
        
        # Remove velocity-encoded frames if present (combined SBO + PCM acquisition)
        if (self.pold_header_slice_info['scanning sequence'] == 1).any():
            self.vel_indx = np.sort(
                list(range(0, self.Phase_deck_tmp.shape[2] - 1, 4)) + 
                list(range(1, self.Phase_deck_tmp.shape[2] - 1, 4))
            )
            self.Phase_deck = np.delete(self.Phase_deck_tmp, self.vel_indx, 2)
        else:
            self.Phase_deck = self.Phase_deck_tmp
        
        # Split into echo 1 and echo 2 images and compute difference
        self.Echo1_deck = self.Phase_deck[:, :, range(0, self.Phase_deck.shape[2], 2)]
        self.Echo2_deck = self.Phase_deck[:, :, range(1, self.Phase_deck.shape[2], 2)]
        self.Echo_diff_deck = self.Echo2_deck - self.Echo1_deck
        self.ndix = self.Echo1_deck.shape
        
        # Calculate echo time difference (convert ms to s)
        self.DeltaTE = (self.pold_header_slice_info['echo_time'].max() - 
                        self.pold_header_slice_info['echo_time'].min()) / 1000

    
    def set_new_data(self):
        """
        Load a new PAR/REC file via file dialog.
        
        Opens a file selection dialog for the user to choose a new PAR file.
        Reloads all image data and updates the GUI display.
        """
        self.raw_img_filename = filedialog.askopenfilename(
            initialdir="/", title="Select file",
            filetypes=(("PAR files", "*.PAR"), ("all files", "*.*"))
        )        
        
        # Parse PAR header for slice-level metadata
        self.hdr_obj = open(self.raw_img_filename) 
        self.pold_header, self.pold_header_slice_info = nib.parrec.parse_PAR_header(self.hdr_obj)
        
        # Load image data using nibabel
        self.raw_img = nib.load(self.raw_img_filename)
        self.img = nib.as_closest_canonical(self.raw_img)
        self.nifti_file = nib.Nifti1Image(self.img.dataobj, self.img.affine, header=self.img.header)
        self.Venc = self.img.header.general_info.get('phase_enc_velocity')[2]
    
        # Compute phase images from real and imaginary components
        self.Phase_deck_tmp = np.rot90(np.arctan2(
            self.img.dataobj[:, :, 0, self.pold_header_slice_info['image_type_mr'] == 2],
            self.img.dataobj[:, :, 0, self.pold_header_slice_info['image_type_mr'] == 1]
        ))
        
        # Extract and scale modulus image
        self.Mod_deck_tmp = np.rot90(self.img.dataobj[:, :, 0, np.logical_and(
            self.pold_header_slice_info['image_type_mr'] == 0, 
            self.pold_header_slice_info['echo number'] == 1
        )])
        self.Mod_deck = ((self.Mod_deck_tmp / self.Mod_deck_tmp.max()) * 2 * math.pi) - math.pi
        
        # Remove velocity-encoded frames
        self.vel_indx = np.sort(
            list(range(0, self.Phase_deck_tmp.shape[2] - 1, 4)) + 
            list(range(1, self.Phase_deck_tmp.shape[2] - 1, 4))
        )
        self.Phase_deck = np.delete(self.Phase_deck_tmp, self.vel_indx, 2)
        
        # Split into echo 1 and echo 2 images and compute difference
        self.Echo1_deck = self.Phase_deck[:, :, range(0, self.Phase_deck.shape[2], 2)]
        self.Echo2_deck = self.Phase_deck[:, :, range(1, self.Phase_deck.shape[2], 2)]
        self.Echo_diff_deck = self.Echo2_deck - self.Echo1_deck
        self.ndix = self.Echo1_deck.shape    
        self.DeltaTE = (self.pold_header_slice_info['echo_time'].max() - 
                        self.pold_header_slice_info['echo_time'].min()) / 1000
  
        # Update GUI display
        global Disp_image_str
        global colormap_str
        change_image(Disp_image_str, colormap_str)
        save_str.set(Img_data.raw_img_filename[0:-4] + '_HbO2_data.csv')


# =============================================================================
# NIFTI IMAGE DATA CLASS
# =============================================================================

class Img_data_nii():
    """
    Class to handle NIfTI format SBO image data (e.g., from Siemens scanners).
    
    Loads and processes dual-echo gradient-echo MRI data from NIfTI files,
    typically converted from DICOM using dcm2niix. Requires four separate NIfTI
    files corresponding to the modulus and phase images at each echo time, each 
    with accompanying JSON sidecar files containing metadata.
    
    The JSON sidecar is used to determine the ImageType ('M' for modulus, 'P' 
    for phase) and EchoNumber (1 or 2) for each file, allowing automatic 
    identification and assignment of the correct data arrays.
    
    Attributes
    ----------
    Echo1_deck : ndarray
        3D phase image array from echo 1 (x, y, frames)
    Echo2_deck : ndarray
        3D phase image array from echo 2 (x, y, frames)
    Echo_diff_deck : ndarray
        3D inter-echo phase difference (Echo2 - Echo1)
    Mod_deck : ndarray
        3D modulus image array, scaled to [0, π] range
    DeltaTE : float
        Echo time difference in seconds
    ndix : tuple
        Shape of the echo image arrays
    """
    def __init__(self, raw_img_filename):
        # Load all four NIfTI files and their JSON sidecars
        # File a
        self.raw_img = nib.load(raw_img_filename[0])
        self.raw_img_filename = raw_img_filename[0]
        self.img = nib.as_closest_canonical(self.raw_img)
        self.nifti_file = nib.Nifti1Image(self.img.dataobj, self.img.affine, header=self.img.header)
        fname_json_tmp = open(os.path.splitext(raw_img_filename[0])[0] + '.json')
        self.img.json_header = json.load(fname_json_tmp)     
        ImageType_a = self.img.json_header['ImageType'][2]
        echo_number_a = self.img.json_header['EchoNumber']
        
        # File b
        self.raw_img_b = nib.load(raw_img_filename[1])
        self.raw_img_filename_b = raw_img_filename[1]
        self.img_b = nib.as_closest_canonical(self.raw_img_b)
        self.nifti_file_b = nib.Nifti1Image(self.img_b.dataobj, self.img_b.affine, header=self.img_b.header)
        fname_json_tmp = open(os.path.splitext(raw_img_filename[1])[0] + '.json')
        self.img_b.json_header = json.load(fname_json_tmp)     
        ImageType_b = self.img_b.json_header['ImageType'][2]
        echo_number_b = self.img_b.json_header['EchoNumber']
        
        # File c
        self.raw_img_c = nib.load(raw_img_filename[2])
        self.raw_img_filename_c = raw_img_filename[2]
        self.img_c = nib.as_closest_canonical(self.raw_img_c)
        self.nifti_file_c = nib.Nifti1Image(self.img_c.dataobj, self.img_c.affine, header=self.img_c.header)
        fname_json_tmp = open(os.path.splitext(raw_img_filename[2])[0] + '.json')
        self.img_c.json_header = json.load(fname_json_tmp)     
        ImageType_c = self.img_c.json_header['ImageType'][2]
        echo_number_c = self.img_c.json_header['EchoNumber']
        
        # File d
        self.raw_img_d = nib.load(raw_img_filename[3])
        self.raw_img_filename_d = raw_img_filename[3]
        self.img_d = nib.as_closest_canonical(self.raw_img_d)
        self.nifti_file_d = nib.Nifti1Image(self.img_d.dataobj, self.img_d.affine, header=self.img_d.header)
        fname_json_tmp = open(os.path.splitext(raw_img_filename[3])[0] + '.json')
        self.img_d.json_header = json.load(fname_json_tmp)     
        ImageType_d = self.img_d.json_header['ImageType'][2]
        echo_number_d = self.img_d.json_header['EchoNumber']

        # Determine which file corresponds to which image type and echo
        self.ImageType = [ImageType_a, ImageType_b, ImageType_c, ImageType_d]
        self.EchoNumber = [echo_number_a, echo_number_b, echo_number_c, echo_number_d]
        
        # Find indices for modulus (M) and phase (P) at each echo
        indx_M = [index for index, value in enumerate(self.ImageType) if value == 'M']
        indx_P = [index for index, value in enumerate(self.ImageType) if value == 'P']
        indx_ec1 = [index for index, value in enumerate(self.EchoNumber) if value == 1]
        indx_ec2 = [index for index, value in enumerate(self.EchoNumber) if value == 2]
        
        def common(a, b): 
            """Find common elements between two lists."""
            c = [value for value in a if value in b] 
            return c
        
        M_ec1_ind = common(indx_M, indx_ec1)[0]   # Modulus, echo 1
        P_ec1_ind = common(indx_P, indx_ec1)[0]   # Phase, echo 1
        P_ec2_ind = common(indx_P, indx_ec2)[0]   # Phase, echo 2
        
        # Extract echo times from the correct JSON headers
        nifti_files_list = [self.img, self.img_b, self.img_c, self.img_d]
        echo1_file_idx = self.EchoNumber.index(1)
        echo2_file_idx = self.EchoNumber.index(2)
        
        if echo1_file_idx == 0:
            self.Echotime1 = self.img.json_header['EchoTime']
        elif echo1_file_idx == 1:
            self.Echotime1 = self.img_b.json_header['EchoTime']
        elif echo1_file_idx == 2:
            self.Echotime1 = self.img_c.json_header['EchoTime']
            
        if echo2_file_idx == 0:
            self.Echotime2 = self.img.json_header['EchoTime']
        elif echo2_file_idx == 1:
            self.Echotime2 = self.img_b.json_header['EchoTime']
        elif echo2_file_idx == 2:
            self.Echotime2 = self.img_c.json_header['EchoTime']
        
        # Calculate echo time difference (ΔTE)
        self.DeltaTE = self.Echotime2 - self.Echotime1

        # Assign modulus data from the correct NIfTI file, scaled to [0, π]
        nifti_list = [self.nifti_file, self.nifti_file_b, self.nifti_file_c, self.nifti_file_d]
        if M_ec1_ind == 0:
            self.Mod_deck = np.squeeze(np.rot90(self.nifti_file.dataobj / self.nifti_file.dataobj.max())) * np.pi
        elif M_ec1_ind == 1:
            self.Mod_deck = np.squeeze(np.rot90(self.nifti_file_b.dataobj / self.nifti_file_b.dataobj.max())) * np.pi
        elif M_ec1_ind == 2:
            self.Mod_deck = np.squeeze(np.rot90(self.nifti_file_c.dataobj / self.nifti_file_c.dataobj.max())) * np.pi
        elif M_ec1_ind == 3:
            self.Mod_deck = np.squeeze(np.rot90(self.nifti_file_d.dataobj / self.nifti_file_d.dataobj.max())) * np.pi
 
        # Assign echo 1 phase data, scaled by 1/2048
        if P_ec1_ind == 0:
            self.Echo1_deck = np.squeeze(np.rot90(self.nifti_file.dataobj)) / 2048
        elif P_ec1_ind == 1:
            self.Echo1_deck = np.squeeze(np.rot90(self.nifti_file_b.dataobj)) / 2048
        elif P_ec1_ind == 2:
            self.Echo1_deck = np.squeeze(np.rot90(self.nifti_file_c.dataobj)) / 2048
        elif P_ec1_ind == 3:
            self.Echo1_deck = np.squeeze(np.rot90(self.nifti_file_d.dataobj)) / 2048
            
        # Assign echo 2 phase data, scaled by 1/2048
        if P_ec2_ind == 0:
            self.Echo2_deck = np.squeeze(np.rot90(self.nifti_file.dataobj)) / 2048
        elif P_ec2_ind == 1:
            self.Echo2_deck = np.squeeze(np.rot90(self.nifti_file_b.dataobj)) / 2048
        elif P_ec2_ind == 2:
            self.Echo2_deck = np.squeeze(np.rot90(self.nifti_file_c.dataobj)) / 2048
        elif P_ec2_ind == 3:
            self.Echo2_deck = np.squeeze(np.rot90(self.nifti_file_d.dataobj)) / 2048

        # Compute inter-echo phase difference
        self.Echo_diff_deck = self.Echo2_deck - self.Echo1_deck
        self.ndix = self.Echo1_deck.shape

    def set_new_data(self):
        """Load new NIfTI data (placeholder for future implementation)."""
        print('NIfTI reload not yet implemented')


# =============================================================================
# INSTANTIATE IMAGE DATA
# =============================================================================
if not nifti_files:
    Img_data = Img_data_parrec(raw_img_filename[0])        

if nifti_files:
    Img_data = Img_data_nii(raw_img_filename)  
        

# =============================================================================
# GLOBAL STATE VARIABLES
# =============================================================================
# Module-level state variables for image display
Disp_image_str = '1echo'    # Current image type: '1echo', '2echo', 'mod', 'Echo_diff', 'ROI'
colormap_str = 'jet'        # Current colormap: 'jet', 'gray', 'viridis'
imgFrame = 1                # Current frame index (1-based)


# =============================================================================
# IMAGE DISPLAY FUNCTIONS
# =============================================================================

def displayed_image(disp_name):
    """
    Return the image data array corresponding to the selected display type.
    
    Parameters
    ----------
    disp_name : str
        Image type identifier: '1echo', '2echo', 'mod', 'Echo_diff', or 'ROI'
    
    Returns
    -------
    ndarray
        3D image array (x, y, frames) for the selected display type
    """
    if disp_name == '1echo':
        return Img_data.Echo1_deck
    elif disp_name == '2echo':
        return Img_data.Echo2_deck
    elif disp_name == 'mod':
        return Img_data.Mod_deck
    elif disp_name == 'Echo_diff':
        return Img_data.Echo2_deck - Img_data.Echo1_deck
    elif disp_name == 'ROI':
        return Sinus_ROI.BWMask * Img_data.Venc * 0.5

Disp_image = displayed_image(Disp_image_str)


def change_image(image_str, cmp_str):
    """
    Update the displayed image with the selected image type and colormap.
    
    Parameters
    ----------
    image_str : str
        Image type identifier
    cmp_str : str
        Colormap name ('jet', 'gray', 'viridis')
    """
    Disp_image = displayed_image(image_str)
    POLD_plot.set_data(Disp_image[:, :, int(imgFrame - 1)])
    POLD_plot.set_cmap(cmp_str)
    POLD_plot.set_clim(climits.lims[0], climits.lims[1])
    canvas.draw()
    return Disp_image_str


# =============================================================================
# IMAGE TYPE SWITCHING FUNCTIONS
# =============================================================================

def change_image_type_str_1echo(self=''):
    """Switch display to Echo 1 phase image."""
    global Disp_image_str
    Disp_image_str = '1echo'
    change_image(Disp_image_str, colormap_str)
    
def change_image_type_str_2echo(self=''):
    """Switch display to Echo 2 phase image."""
    global Disp_image_str
    Disp_image_str = '2echo'
    change_image(Disp_image_str, colormap_str)

def change_image_type_str_mod(self=''):
    """Switch display to modulus image."""
    global Disp_image_str
    Disp_image_str = 'mod'
    change_image(Disp_image_str, colormap_str)

def change_image_type_str_echo_diff(self=''):
    """Switch display to inter-echo phase difference image (Echo 2 - Echo 1)."""
    global Disp_image_str
    Disp_image_str = 'Echo_diff'
    change_image(Disp_image_str, colormap_str)
    
def change_image_type_str_ROI(self=''):
    """Switch display to ROI mask overlay."""
    global Disp_image_str
    Disp_image_str = 'ROI'
    change_image(Disp_image_str, colormap_str)


# =============================================================================
# COLORMAP SWITCHING FUNCTIONS
# =============================================================================

def change_cmap_jet(self='jet'):
    """Switch colormap to jet."""
    global colormap_str
    colormap_str = 'jet'
    change_image(Disp_image_str, colormap_str)

def change_cmap_gray(self='gray'):
    """Switch colormap to grayscale."""
    global colormap_str
    colormap_str = 'gray'
    change_image(Disp_image_str, colormap_str)

def change_cmap_viridis(self='viridis'):
    """Switch colormap to viridis."""
    global colormap_str
    colormap_str = 'viridis'
    change_image(Disp_image_str, colormap_str)

    
def popup_change_colorbar():
    """Open a popup window for manually adjusting colorbar intensity limits."""
    popup_change_colorbar = Tk()
    popup_change_colorbar.title('Colorbar limits')
    GUI_width = window.winfo_screenwidth() * 0.10
    GUI_height = window.winfo_screenheight() * 0.22
    popup_change_colorbar.geometry(str(int(GUI_width)) + 'x' + str(int(GUI_height)))
    popup_change_colorbar.resizable(True, True)

    Min_entry_text = Label(popup_change_colorbar, text="Min:", width=7, padx=0)
    Min_entry_text.grid(row=1, rowspan=1, column=0, columnspan=1, sticky='n', pady=10, padx=0)
    Min_entry = Entry(popup_change_colorbar, width=7)
    Min_entry.grid(row=1, rowspan=1, column=1, columnspan=1, sticky='nw', pady=10, padx=0)
    Min_entry.insert(END, '-3.140')
    
    Max_entry_text = Label(popup_change_colorbar, text="Max:", width=7, padx=0)
    Max_entry_text.grid(row=0, rowspan=1, column=0, columnspan=1, sticky='n', pady=0, padx=0)
    Max_entry = Entry(popup_change_colorbar, width=7)
    Max_entry.grid(row=0, rowspan=1, column=1, columnspan=1, sticky='nw', pady=0, padx=0)
    Max_entry.insert(END, '3.140')
    
    def update_colorbar():
        climits.lims = [float(Min_entry.get()), float(Max_entry.get())]
        change_image(Disp_image_str, colormap_str)
    
    UpdateCB_button = Button(master=popup_change_colorbar, height=2, width=9,
                             text="Update Colorbar", command=update_colorbar)
    UpdateCB_button.grid(row=2, rowspan=1, column=0, columnspan=2, sticky='nw', pady=2, padx=10)
    

# =============================================================================
# TOP MENU BAR
# =============================================================================
menubar = Menu(window)
window.config(menu=menubar)

# File menu
analysismenu = Menu(menubar, tearoff=1)
analysis_submenu = Menu(analysismenu, tearoff=1)
menubar.add_cascade(label="File", menu=analysismenu)
analysismenu.add_command(label="Load new file", command=Img_data.set_new_data, accelerator="Control+n")
window.bind_all('<Control-Key-n>', func=Img_data.set_new_data)

# Image type submenu
imagemenu = Menu(menubar, tearoff=1)
submenu = Menu(imagemenu, tearoff=1)
menubar.add_cascade(label="Image", menu=imagemenu)
imagemenu.add_cascade(label="Change image type", menu=submenu)

submenu.add_radiobutton(label="Echo 1", accelerator="Control+1", command=change_image_type_str_1echo)
submenu.add_radiobutton(label="Echo 2", accelerator="Control+2", command=change_image_type_str_2echo)
submenu.add_radiobutton(label="Echo 2 - Echo 1", accelerator="Control+3", command=change_image_type_str_echo_diff)
submenu.add_radiobutton(label="Modulus", accelerator="Control+4", command=change_image_type_str_mod)
submenu.add_radiobutton(label="ROI mask", accelerator="Control+5", command=change_image_type_str_ROI)

window.bind_all('<Control-Key-1>', func=change_image_type_str_1echo)
window.bind_all('<Control-Key-2>', func=change_image_type_str_2echo)
window.bind_all('<Control-Key-3>', func=change_image_type_str_echo_diff)
window.bind_all('<Control-Key-4>', func=change_image_type_str_mod)
window.bind_all('<Control-Key-5>', func=change_image_type_str_ROI)

# Colormap submenu
subsubmenu = Menu(submenu, tearoff=1)
imagemenu.add_cascade(label="Change colorbar", menu=subsubmenu)
subsubmenu.add_command(label="Jet", accelerator="Control+q", command=change_cmap_jet)
subsubmenu.add_command(label="Gray", accelerator="Control+w", command=change_cmap_gray)
subsubmenu.add_command(label="Viridis", accelerator="Control+e", command=change_cmap_viridis)
subsubmenu.add_command(label="Change colorbar limit", accelerator="Control+l", command=popup_change_colorbar)

window.bind_all('<Control-Key-q>', func=change_cmap_jet)
window.bind_all('<Control-Key-w>', func=change_cmap_gray)
window.bind_all('<Control-Key-e>', func=change_cmap_viridis)


# =============================================================================
# GUI LAYOUT
# =============================================================================
Grid.columnconfigure(window, 1, weight=1)
Grid.columnconfigure(window, 0, weight=1)
Grid.columnconfigure(window, 5, weight=1)

# Create main figure for displaying MRI image
fig = plt.figure(figsize=(3.3, 3.3), dpi=100)
ax = fig.add_subplot(111)

class climits():
    """Colorbar intensity limits for image display."""
    lims = [-3.14, 3.14]

POLD_plot = ax.imshow(Disp_image[:, :, imgFrame - 1], cmap=plt.get_cmap(colormap_str),
                      vmin=climits.lims[0], vmax=climits.lims[1], interpolation='none')
fig.colorbar(POLD_plot, ax=ax, shrink=0.6)

ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()

# ROI overlay line plots (Sinus=white, Tissue=blue, Unwrap=cyan)
Sinus_ROI_line_plot, = ax.plot([], [], '.w-')
Tissue_ROI_line_plot, = ax.plot([], [], '.b-')
Unwrap_ROI_line_plot, = ax.plot([], [], '.c-')

canvas = FigureCanvasTkAgg(fig, master=window)
canvas.draw()
canvas.get_tk_widget().grid(row=1, rowspan=1, column=1, columnspan=1, padx=0, pady=0)

# Create HbO2 results plot
fig_flow = plt.figure(figsize=(1.8, 1.8), dpi=110)
ax_HbO2 = fig_flow.add_subplot(111)

ax_HbO2.set_position([0.2, 0.15, 0.7, 0.7])
ax_HbO2.tick_params(labelsize=7)
ax_HbO2.set_ylabel('HbO2', fontsize=8.0)
ax_HbO2.set_xlabel('Index', fontsize=8.0)
HbO2_line_plot, = ax_HbO2.plot([], [], '.r-')

canvas_flow = FigureCanvasTkAgg(fig_flow, master=window)
canvas_flow.draw()
canvas_flow.get_tk_widget().grid(row=1, rowspan=1, column=5, columnspan=1, padx=20, pady=50)

# Navigation toolbar
ToolbarFrame = Frame(window)
ToolbarFrame.grid(row=2, column=1, rowspan=1, columnspan=1, padx=0, pady=0, sticky='nw')
toobar = NavigationToolbar2Tk(canvas, ToolbarFrame)


# =============================================================================
# FRAME NAVIGATION
# =============================================================================

def update_image(self):
    """
    Update the displayed image and ROI overlays when navigating frames.
    
    Parameters
    ----------
    self : str or int
        Frame number (from slider or keyboard navigation)
    """
    global imgFrame
    imgFrame = int(self)
    Disp_image = displayed_image(Disp_image_str)
    POLD_plot.set_data(Disp_image[:, :, int(imgFrame - 1)])
    
    # Update sinus ROI overlay
    Sinus_ROI_as_array = np.array(Sinus_ROI.polygon[int(imgFrame - 1)])
    if np.equal(Sinus_ROI_as_array, None).all():
        Sinus_ROI_line_plot.set_ydata([])
        Sinus_ROI_line_plot.set_xdata([])
    else:
        ROI_as_array_tmp = np.append(Sinus_ROI_as_array[:, :], Sinus_ROI_as_array[0, :]).reshape(
            Sinus_ROI_as_array.shape[0] + 1, Sinus_ROI_as_array.shape[1])
        Sinus_ROI_line_plot.set_ydata(ROI_as_array_tmp[:, 1])
        Sinus_ROI_line_plot.set_xdata(ROI_as_array_tmp[:, 0])
    
    # Update tissue ROI overlay
    Tissue_ROI_as_array = np.array(Tissue_ROI.polygon[int(imgFrame - 1)])
    if np.equal(Tissue_ROI_as_array, None).all():
        Tissue_ROI_line_plot.set_ydata([])
        Tissue_ROI_line_plot.set_xdata([])
    else:
        Tissue_ROI_as_array_tmp = np.append(Tissue_ROI_as_array[:, :], Tissue_ROI_as_array[0, :]).reshape(
            Tissue_ROI_as_array.shape[0] + 1, Tissue_ROI_as_array.shape[1])
        Tissue_ROI_line_plot.set_ydata(Tissue_ROI_as_array_tmp[:, 1])
        Tissue_ROI_line_plot.set_xdata(Tissue_ROI_as_array_tmp[:, 0])
        
    canvas.draw()    
    return imgFrame


# Frame slider
slider_scale = Scale(window, to=1, from_=Img_data.Echo1_deck.shape[2], width=20, length=400, command=update_image)
slider_scale.grid(row=1, rowspan=1, column=4, columnspan=1, padx=0, pady=0, sticky='w')


def key_arrow_up(self=''):
    """Navigate to the next frame (arrow up key)."""
    global imgFrame
    if imgFrame > (Img_data.Echo1_deck.shape[2] - 1):
        imgFrame = imgFrame
    else:
        imgFrame = imgFrame + 1
    update_image(imgFrame)
    slider_scale.set(imgFrame)

window.bind_all('<Up>', func=key_arrow_up)       
    
def key_arrow_down(self=''):
    """Navigate to the previous frame (arrow down key)."""
    global imgFrame
    if imgFrame < 2:
        imgFrame = imgFrame
    else:
        imgFrame = imgFrame - 1
    update_image(imgFrame)
    slider_scale.set(imgFrame)

window.bind_all('<Down>', func=key_arrow_down)    


# Headline text
headline_text = Label(window, text="Draw ROI to calculate oxygen saturation in the blood vessel")
headline_text.grid(row=0, rowspan=1, column=0, columnspan=2, sticky='nw')


# =============================================================================
# ROI DATA CLASS
# =============================================================================

class ROI:
    """
    Region of Interest (ROI) data container.
    
    Stores polygon vertices, binary masks, and flags for each frame. 
    Used for both the sinus (vessel) ROI and the tissue ROI.
    
    Attributes
    ----------
    polygon : list
        List of polygon vertex arrays, one per frame (None if undefined)
    BWMask : ndarray
        3D binary mask array matching image dimensions
    flag : list
        List of flags (0/1) indicating which frames have defined ROIs
    """
    def __init__(self):
        self.polygon = [None] * Img_data.Echo1_deck.shape[2]
        self.BWMask = Img_data.Echo1_deck * False
        self.flag = [0] * (Img_data.ndix[2])

    def set_polygon(self, verts, curr_frame):
        """
        Set ROI polygon for a specific frame and compute the binary mask.
        
        Parameters
        ----------
        verts : list of tuples
            Polygon vertex coordinates [(x1,y1), (x2,y2), ...]
        curr_frame : int
            Frame number (1-based)
        """
        self.polygon[int(curr_frame - 1)] = verts
        self.flag[int(curr_frame - 1)] = 1
        
        # Convert polygon to binary mask using matplotlib Path
        path = Path(verts)
        x, y = np.meshgrid(np.arange(Img_data.ndix[0]), np.arange(Img_data.ndix[1]))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T
        grid = path.contains_points(points)
        grid = grid.reshape((Img_data.ndix[0], Img_data.ndix[1]))        
        self.BWMask[:, :, int(curr_frame - 1)] = grid

    def loaded_roi_from_file(Loaded_ROI_data):
        """Load ROI data from a previously saved file."""
        self.polygon = Loaded_ROI_data['PCMROI_poly'].tolist()
        self.BWMask = Loaded_ROI_data['PCMROI_BWMask']
        self.flag = Loaded_ROI_data['PCMROI_flag']
        global imgFrame
        update_image(imgFrame)

# Instantiate the three ROI objects        
Sinus_ROI = ROI()    # Vessel (sinus) ROI
Tissue_ROI = ROI()   # Surrounding tissue ROI
Unwrap_ROI = ROI()   # Phase unwrapping ROI


# =============================================================================
# ROI POLYGON DRAWING CLASS
# =============================================================================

class ROIPolygon(object):
    """
    Interactive polygon drawing tool for ROI delineation.
    
    Wraps matplotlib's PolygonSelector to enable interactive polygon drawing 
    on the image. The drawn polygon is automatically converted to a binary 
    mask and stored in the appropriate ROI object.
    
    Parameters
    ----------
    name : str
        ROI identifier: 'SS' (sinus), 'Tissue', or 'Unwrap'
    ax : matplotlib.axes.Axes
        Axes to draw the polygon on
    row : int
        Image row dimension
    col : int
        Image column dimension
    """
    def __init__(self, name, ax, row, col):
        self.name = name
        self.canvas = ax.figure.canvas
        self.polygon = ''
        self.PS = PolygonSelector(ax, self.onselect,
                                  props=dict(color='k', alpha=0.5),
                                  handle_props=dict(mec='k', mfc='k', alpha=0.5, markersize=1),
                                  grab_range=20)
        self.PS.set_visible(False)
        self.PS.set_active(False)

    def onselect(self, verts):
        """
        Callback triggered when a polygon is completed.
        
        Sets the polygon on the appropriate ROI object and updates the 
        display overlay.
        """
        self.canvas.draw_idle()
        self.polygons = verts
        
        if self.name == 'SS':
            Sinus_ROI.set_polygon(verts, imgFrame)
            Sinus_ROI.flag[imgFrame - 1] = 1
            Sinus_ROI_as_array = np.array(self.PS.verts)
            Sinus_ROI_as_array_tmp = np.append(Sinus_ROI_as_array[:, :], Sinus_ROI_as_array[0, :]).reshape(
                Sinus_ROI_as_array.shape[0] + 1, Sinus_ROI_as_array.shape[1])
            Sinus_ROI_line_plot.set_ydata(Sinus_ROI_as_array_tmp[:, 1])
            Sinus_ROI_line_plot.set_xdata(Sinus_ROI_as_array_tmp[:, 0])
            global Disp_image_str
            global colormap_str
            change_image(Disp_image_str, colormap_str)
    
        if self.name == 'Tissue':
            Tissue_ROI.set_polygon(verts, imgFrame)
            Tissue_ROI.flag[imgFrame - 1] = 1
            Tissue_ROI_as_array = np.array(self.PS.verts)
            Tissue_ROI_as_array_tmp = np.append(Tissue_ROI_as_array[:, :], Tissue_ROI_as_array[0, :]).reshape(
                Tissue_ROI_as_array.shape[0] + 1, Tissue_ROI_as_array.shape[1])
            Tissue_ROI_line_plot.set_ydata(Tissue_ROI_as_array_tmp[:, 1])
            Tissue_ROI_line_plot.set_xdata(Tissue_ROI_as_array_tmp[:, 0])
            change_image(Disp_image_str, colormap_str)

        if self.name == 'Unwrap':
            Unwrap_ROI.set_polygon(verts, imgFrame)
            Unwrap_ROI.flag[imgFrame - 1] = 1
            Unwrap_ROI_as_array = np.array(self.PS.verts)
            Unwrap_ROI_as_array_tmp = np.append(Unwrap_ROI_as_array[:, :], Unwrap_ROI_as_array[0, :]).reshape(
                Unwrap_ROI_as_array.shape[0] + 1, Unwrap_ROI_as_array.shape[1])
            Unwrap_ROI_line_plot.set_ydata(Unwrap_ROI_as_array_tmp[:, 1])
            Unwrap_ROI_line_plot.set_xdata(Unwrap_ROI_as_array_tmp[:, 0])
            change_image(Disp_image_str, colormap_str)

        self.canvas.draw_idle()


# =============================================================================
# SINUS ROI CONTROLS
# =============================================================================

class AddSinusROI():
    """Controls for adding and managing the sinus (vessel) ROI."""
    def add_roi():
        """Activate polygon drawing tool for sinus ROI."""
        if hasattr(AddSinusROI, 'PS_SS'):
            AddSinusROI.PS_SS.PS.set_visible(False)
            AddSinusROI.PS_SS.PS.set_active(False)
        AddSinusROI.PS_SS = ROIPolygon('SS', POLD_plot.axes, Img_data.ndix[0], Img_data.ndix[1])
        AddSinusROI.PS_SS.PS.set_visible(True)
        AddSinusROI.PS_SS.PS.set_active(True)
        UpdateSinusROI_button.configure(text='Stop ROI edit')
        canvas.draw() 

def CopySinusROI():
    """Copy the current sinus ROI to all frames."""
    first_indx = Sinus_ROI.flag.index(1)
    if Sinus_ROI.flag[int(imgFrame - 1)] == 1:
        for x in range(len(Sinus_ROI.polygon)):
            Sinus_ROI.polygon[x] = Sinus_ROI.polygon[int(imgFrame - 1)]
            Sinus_ROI.BWMask[:, :, x] = Sinus_ROI.BWMask[:, :, int(imgFrame - 1)]
            Sinus_ROI.flag[x] = 1
    else:
        for x in range(len(Sinus_ROI.polygon)): 
            Sinus_ROI.polygon[x] = Sinus_ROI.polygon[first_indx]
            Sinus_ROI.BWMask[:, :, x] = Sinus_ROI.BWMask[:, :, first_indx]
            Sinus_ROI.flag[x] = 1

def UpdateSinusROI():
    """Toggle sinus ROI between editable and locked states."""
    if AddSinusROI.PS_SS.PS.get_active():
        AddSinusROI.PS_SS.PS.set_visible(False)
        AddSinusROI.PS_SS.PS.set_active(False)
        canvas.draw() 
        UpdateSinusROI_button.configure(text='Edit ROI')
    else:
        AddSinusROI.PS_SS.PS.set_visible(True)
        AddSinusROI.PS_SS.PS.set_active(True)
        canvas.draw() 
        UpdateSinusROI_button.configure(text='Stop ROI edit')


# =============================================================================
# TISSUE ROI CONTROLS
# =============================================================================

class AddTissueROI():
    """Controls for adding and managing the tissue ROI."""
    def add_roi():
        """Activate polygon drawing tool for tissue ROI."""
        if hasattr(AddTissueROI, 'PS_tissue'):
            AddTissueROI.PS_tissue.PS.set_visible(False)
            AddTissueROI.PS_tissue.PS.set_active(False)
        AddTissueROI.PS_tissue = ROIPolygon('Tissue', POLD_plot.axes, Img_data.ndix[0], Img_data.ndix[1])
        AddTissueROI.PS_tissue.PS.set_visible(True)
        AddTissueROI.PS_tissue.PS.set_active(True)
        canvas.draw() 
        UpdateTissueROI_button.configure(text='Stop ROI edit')

def CopyTissueROI():
    """Copy the current tissue ROI to all frames."""
    first_indx = Tissue_ROI.flag.index(1)
    if Tissue_ROI.flag[int(imgFrame - 1)] == 1:
        for x in range(len(Tissue_ROI.polygon)):
            Tissue_ROI.polygon[x] = Tissue_ROI.polygon[int(imgFrame - 1)]
            Tissue_ROI.BWMask[:, :, x] = Tissue_ROI.BWMask[:, :, int(imgFrame - 1)]
            Tissue_ROI.flag[x] = 1
    else:
        for x in range(len(Tissue_ROI.polygon)): 
            Tissue_ROI.polygon[x] = Tissue_ROI.polygon[first_indx]
            Tissue_ROI.BWMask[:, :, x] = Tissue_ROI.BWMask[:, :, first_indx]
            Tissue_ROI.flag[x] = 1

def UpdateTissueROI():
    """Toggle tissue ROI between editable and locked states."""
    if AddTissueROI.PS_tissue.PS.get_active():
        AddTissueROI.PS_tissue.PS.set_visible(False)
        AddTissueROI.PS_tissue.PS.set_active(False)
        canvas.draw() 
        UpdateTissueROI_button.configure(text='Edit ROI')
    else:
        AddTissueROI.PS_tissue.PS.set_visible(True)
        AddTissueROI.PS_tissue.PS.set_active(True)
        canvas.draw() 
        UpdateTissueROI_button.configure(text='Stop ROI edit')


# =============================================================================
# UNWRAP ROI CONTROLS
# =============================================================================

class AddUnwrapROI():
    """Controls for adding and managing the phase unwrap ROI."""
    def add_roi():
        """Activate polygon drawing tool for unwrap ROI."""
        if hasattr(AddUnwrapROI, 'PS_tissue'):
            AddUnwrapROI.PS_unwrap.PS.set_visible(False)
            AddUnwrapROI.PS_unwrap.PS.set_active(False)
        AddUnwrapROI.PS_unwrap = ROIPolygon('Unwrap', POLD_plot.axes, Img_data.ndix[0], Img_data.ndix[1])
        AddUnwrapROI.PS_unwrap.PS.set_visible(True)
        AddUnwrapROI.PS_unwrap.PS.set_active(True)
        canvas.draw() 
        UpdateUnwrapROI_button.configure(text='Stop ROI edit')

def CopyUnwrapROI():
    """Copy the current unwrap ROI to all frames."""
    first_indx = Unwrap_ROI.flag.index(1)
    if Unwrap_ROI.flag[int(imgFrame - 1)] == 1:
        for x in range(len(Unwrap_ROI.polygon)):
            Unwrap_ROI.polygon[x] = Unwrap_ROI.polygon[int(imgFrame - 1)]
            Unwrap_ROI.BWMask[:, :, x] = Unwrap_ROI.BWMask[:, :, int(imgFrame - 1)]
            Unwrap_ROI.flag[x] = 1
    else:
        for x in range(len(Unwrap_ROI.polygon)): 
            Unwrap_ROI.polygon[x] = Unwrap_ROI.polygon[first_indx]
            Unwrap_ROI.BWMask[:, :, x] = Unwrap_ROI.BWMask[:, :, first_indx]
            Unwrap_ROI.flag[x] = 1

def UpdateUnwrapROI():
    """Toggle unwrap ROI between editable and locked states."""
    if AddUnwrapROI.PS_unwrap.PS.get_active():
        AddUnwrapROI.PS_unwrap.PS.set_visible(False)
        AddUnwrapROI.PS_unwrap.PS.set_active(False)
        canvas.draw() 
        UpdateUnwrapROI_button.configure(text='ROI edit')
    else:
        AddUnwrapROI.PS_unwrap.PS.set_visible(True)
        AddUnwrapROI.PS_unwrap.PS.set_active(True)
        canvas.draw()
        UpdateUnwrapROI_button.configure(text='Stop ROI edit')

def RemoveUnwrapROI():
    """Remove the unwrap ROI and reset its overlay."""
    Unwrap_ROI = ROI()
    Unwrap_ROI_line_plot.set_ydata([None])
    Unwrap_ROI_line_plot.set_xdata([None])
    change_image(Disp_image_str, colormap_str)
    

# =============================================================================
# PHASE UNWRAPPING
# =============================================================================

def UnwrapImage():
    """
    Apply selective phase unwrapping within the unwrap ROI.
    
    Adds 2π to voxels within the unwrap ROI whose phase values fall below 
    the user-specified threshold. The unwrapping is applied to the currently 
    displayed echo image (Echo 1 or Echo 2), and the echo difference map 
    is recalculated afterwards.
    """
    Unwrap_ROI.BWMask = np.where(Unwrap_ROI.BWMask == 0, float('nan'), Unwrap_ROI.BWMask)
    if Disp_image_str == '1echo':
        EchoWunrap_tmp = Unwrap_ROI.BWMask * Img_data.Echo1_deck
        Voxels_for_unwrap_indx = EchoWunrap_tmp < float(e_unwrap.get())
        Img_data.Echo1_deck[Voxels_for_unwrap_indx] = EchoWunrap_tmp[Voxels_for_unwrap_indx] + math.pi * 2
        Img_data.Echo_diff_deck = Img_data.Echo2_deck - Img_data.Echo1_deck
    if Disp_image_str == '2echo':
        EchoWunrap_tmp = Unwrap_ROI.BWMask * Img_data.Echo2_deck
        Voxels_for_unwrap_indx = EchoWunrap_tmp < float(e_unwrap.get())
        Img_data.Echo2_deck[Voxels_for_unwrap_indx] = EchoWunrap_tmp[Voxels_for_unwrap_indx] + math.pi * 2
        Img_data.Echo_diff_deck = Img_data.Echo2_deck - Img_data.Echo1_deck
    change_image(Disp_image_str, colormap_str)


# =============================================================================
# REGION GROWING ALGORITHM
# =============================================================================

class RegGrow():
    """
    Semi-automatic region-growing algorithm for sinus ROI delineation.
    
    The algorithm starts from a user-selected seed point and iteratively 
    includes neighbouring voxels whose signal values exceed a user-defined 
    threshold, constrained by a maximum distance from the seed point. For 
    multi-frame data, the algorithm tracks the vessel across frames by 
    updating the seed to the voxel with the highest signal in the current region.
    """
    nRow, nCol, nSlice = Img_data.Echo1_deck.shape
    qu = 1
    ginput_input = []
    btm_press_event = []
    
    def create_mask(event):
        """
        Region growing callback triggered by mouse click.
        
        Grows a region from the clicked seed point across all frames using the
        currently displayed image type and user-specified threshold/distance 
        parameters.
        """
        window.config(cursor='')
        ginput_input = event.xdata, event.ydata
        maxDist = int(e1.get())
        ThresVal = float(e2.get()) 
        fig.canvas.callbacks.disconnect(RegGrow.btm_press_event)
        seed_tmp = np.round(ginput_input)
        seed = np.flip(seed_tmp)
        Reg_mask = np.zeros(Img_data.ndix)
        Reg_mask_tmp = np.zeros(Img_data.ndix)

        # Select threshold image based on current display type
        if Disp_image_str == '1echo':
            Thres_image_tmp = Img_data.Echo1_deck
        if Disp_image_str == '2echo':
            Thres_image_tmp = Img_data.Echo2_deck
        if Disp_image_str == 'Echo_diff':
            Thres_image_tmp = Img_data.Echo_diff_deck
        if Disp_image_str == 'mod':
            Thres_image_tmp = Img_data.Mod_deck   

        # Grow region across all frames
        for nn in range(Img_data.ndix[2]):
            queue = seed
            Imax = int(seed[0])
            Jmax = int(seed[1])
            while queue.any():
                if queue.ndim == 1:
                    xv = int(queue[0])
                    yv = int(queue[1])
                else:
                    xv = int(queue[0][0])
                    yv = int(queue[0][1])
                   
                for n in [-1, 0, 1]:
                    for m in [-1, 0, 1]:
                        if (xv + n > 0 and xv + n <= RegGrow.nRow and 
                            yv + m > 0 and yv + m <= RegGrow.nCol and 
                            any([n, m]) and 
                            Reg_mask_tmp[xv + n, yv + m, nn] == 0 and 
                            np.sqrt((xv + n - Imax)**2 + (yv + m - Jmax)**2) < maxDist and 
                            Thres_image_tmp[xv + n, yv + m, nn] >= ThresVal):
                            Reg_mask_tmp[(xv + n, yv + m, nn)] = 1
                            queue = np.vstack((queue, np.array([xv + n, yv + m])))
                
                # Apply binary erosion and update seed for next frame
                Reg_mask[:, :, nn] = scipy.ndimage.binary_erosion(Reg_mask_tmp[:, :, nn], iterations=2).astype(np.float32)
                queue = np.delete(queue, (0), axis=0)
                RegGrow.qu = queue  
                masked_inp_tmp = Img_data.Echo_diff_deck[:, :, nn] * Reg_mask[:, :, nn]
                New_seed = np.where(masked_inp_tmp == np.amax(masked_inp_tmp))
                seed = np.array([New_seed[0][0], New_seed[1][0]])
        
        # Convert region masks to polygon contours
        for mm in range(Img_data.ndix[2]):
            found_contours = measure.find_contours(Reg_mask[:, :, mm], 0.5)
            cnt_length = np.zeros(len(found_contours))
            for bb in range(len(found_contours)):
                cnt_length[bb] = found_contours[bb].shape[0]
            max_count_idx = np.where(cnt_length == np.amax(cnt_length))   
            Sinus_ROI.set_polygon(np.fliplr(found_contours[max_count_idx[0][0]]), mm + 1)
        update_image(imgFrame)

    def Calc_Auto_ROI():
        """Activate seed point selection mode for region growing."""
        window.config(cursor='plus green white')
        RegGrow.btm_press_event = fig.canvas.callbacks.connect('button_press_event', RegGrow.create_mask)


# =============================================================================
# GUI BUTTONS - SINUS ROI
# =============================================================================
SinusROI_button_group = LabelFrame(window, text='Sinus ROI', borderwidth=2, relief='solid')
SinusROI_button_group.grid(row=1, rowspan=1, column=0, columnspan=1, sticky='nw', padx=10, pady=20)

AddSinusROI_button = Button(master=SinusROI_button_group, height=2, width=10,
                            text="Add ROI", command=AddSinusROI.add_roi)
AddSinusROI_button.grid(row=0, rowspan=1, column=0, columnspan=2, sticky='nw', pady=2, padx=10)

UpdateSinusROI_button = Button(master=SinusROI_button_group, height=2, width=10,
                               text="Edit ROI", command=UpdateSinusROI)
UpdateSinusROI_button.grid(row=1, rowspan=1, column=0, columnspan=2, sticky='nw', pady=2, padx=10)

CopySinusROI_button = Button(master=SinusROI_button_group, height=2, width=10,
                             text="Copy ROI to \n all frames", command=CopySinusROI)
CopySinusROI_button.grid(row=3, rowspan=1, column=0, columnspan=2, sticky='nw', pady=2, padx=10)

AutoROI_button = Button(master=SinusROI_button_group, height=3, width=10,
                        text="Automatic \n delineation", command=RegGrow.Calc_Auto_ROI)
AutoROI_button.grid(row=4, rowspan=1, column=0, columnspan=2, sticky='nw', pady=2, padx=10)

Max_dist_text = Label(SinusROI_button_group, text="Max. dist:", width=7, padx=0)
Max_dist_text.grid(row=5, rowspan=1, column=0, columnspan=1, sticky='n', pady=0, padx=0)

e1 = Entry(SinusROI_button_group, width=4)
e1.grid(row=5, rowspan=1, column=1, columnspan=1, sticky='nw', pady=0, padx=0)
e1.insert(END, '7')

Threshold_text = Label(SinusROI_button_group, text="Threshold:")
Threshold_text.grid(row=6, rowspan=1, column=0, columnspan=1, sticky='nw', pady=0, padx=10)

e2 = Entry(SinusROI_button_group, width=4)
e2.grid(row=6, rowspan=1, column=1, columnspan=1, sticky='nw', pady=0, padx=0)
e2.insert(END, '1')


# =============================================================================
# GUI BUTTONS - TISSUE ROI
# =============================================================================
TissueROI_button_group = LabelFrame(window, text='Tissue ROI', borderwidth=2, relief='solid')
TissueROI_button_group.grid(row=1, rowspan=1, column=0, columnspan=1, sticky='sw', padx=10, pady=20)

AddTissueROI_button = Button(master=TissueROI_button_group, height=2, width=10,
                             text="Add ROI", command=AddTissueROI.add_roi)
AddTissueROI_button.grid(row=0, rowspan=1, column=0, columnspan=2, sticky='nw', pady=2, padx=10)

UpdateTissueROI_button = Button(master=TissueROI_button_group, height=2, width=10,
                                text="Edit ROI", command=UpdateTissueROI)
UpdateTissueROI_button.grid(row=1, rowspan=1, column=0, columnspan=2, sticky='nw', pady=2, padx=10)

CopyTissueROI_button = Button(master=TissueROI_button_group, height=2, width=10,
                              text="Copy ROI to \n all frames", command=CopyTissueROI)
CopyTissueROI_button.grid(row=3, rowspan=1, column=0, columnspan=2, sticky='nw', pady=2, padx=10)


# =============================================================================
# GUI BUTTONS - UNWRAP ROI
# =============================================================================
Unwrap_button_group = LabelFrame(window, text='Unwrap image', borderwidth=2, relief='solid')
Unwrap_button_group.grid(row=2, rowspan=1, column=0, columnspan=1, sticky='nw', padx=10, pady=1)

AddUnwrapROI_button = Button(master=Unwrap_button_group, height=2, width=10,
                             text="Add ROI", command=AddUnwrapROI.add_roi)
AddUnwrapROI_button.grid(row=0, rowspan=1, column=0, columnspan=2, sticky='nw', pady=2, padx=10)
 
UpdateUnwrapROI_button = Button(master=Unwrap_button_group, height=2, width=10,
                                text="Edit ROI", command=UpdateUnwrapROI)
UpdateUnwrapROI_button.grid(row=1, rowspan=1, column=0, columnspan=2, sticky='nw', pady=2, padx=10)

CopyUnwrapROI_button = Button(master=Unwrap_button_group, height=2, width=10,
                              text="Copy ROI to \n all frames", command=CopyUnwrapROI)
CopyUnwrapROI_button.grid(row=2, rowspan=1, column=0, columnspan=2, sticky='nw', pady=2, padx=10)

RemoveUnwrapROI_button = Button(master=Unwrap_button_group, height=2, width=10,
                                text="Remove ROI", command=RemoveUnwrapROI)
RemoveUnwrapROI_button.grid(row=3, rowspan=1, column=0, columnspan=2, sticky='nw', pady=2, padx=10)

UnwrapLimit = Label(Unwrap_button_group, text="Unwrap limit:", width=9, padx=0)
UnwrapLimit.grid(row=4, rowspan=1, column=0, columnspan=1, sticky='nw', pady=0, padx=0)

e_unwrap = Entry(Unwrap_button_group, width=4)
e_unwrap.grid(row=4, rowspan=1, column=1, columnspan=1, sticky='ne', pady=0, padx=0)
e_unwrap.insert(END, '0')

Unwrap_button = Button(master=Unwrap_button_group, height=2, width=10,
                       text="Unwrap", command=UnwrapImage)
Unwrap_button.grid(row=5, rowspan=1, column=0, columnspan=2, sticky='nw', pady=2, padx=10)


# =============================================================================
# HbO2 CALCULATION
# =============================================================================

class HbO2_output:
    """
    Oxygen saturation (HbO2) calculation engine.
    
    Implements the susceptibility-based oximetry equation to calculate blood 
    oxygen saturation from the inter-echo phase difference in vessel and 
    tissue ROIs.
    
    The SBO equation:
        HbO2 = 1 - 2*(Δφ_sinus - Δφ_tissue) / 
                (γ * 4π * Δχ_do * B0 * (cos²θ - 1/3) * Hct * ΔTE) - Δχ_0
    
    where:
        γ = 267.513 MHz/T (gyromagnetic ratio)
        Δχ_do = 0.27 ppm (susceptibility of fully deoxygenated blood)
        B0 = 3T (main magnetic field strength)
        Δχ_0 = 0.0296 ppm (oxy-haemoglobin susceptibility offset)
    
    Attributes
    ----------
    Hct : float
        Hematocrit value (default: 0.43)
    SS_angle : float
        Vessel angle relative to B0 in degrees (default: 0)
    HbO2 : ndarray
        Calculated oxygen saturation for each frame
    Sinus_mean_timeseries : ndarray
        Mean inter-echo phase difference in sinus ROI per frame
    Tissue_mean_timeseries : ndarray
        Mean inter-echo phase difference in tissue ROI per frame
    """
    def __init__():
        if args.hct_input is None:
            HbO2_output.Hct = 0.43
        else: 
            HbO2_output.Hct = args.hct_input
        
        if args.ss_angle_input is None:
            HbO2_output.SS_angle = 0
        else: 
            HbO2_output.SS_angle = args.ss_angle_input
        
    def Calc_HbO2():
        """
        Calculate blood oxygen saturation (HbO2) from the ROI data.
        
        Computes the mean inter-echo phase difference in the sinus and tissue 
        ROIs, then applies the SBO equation to derive HbO2 for each frame.
        Results are displayed in the HbO2 plot panel.
        """
        # Calculate mean phase difference in sinus ROI
        Sinus_ROI_tmp = np.where(Sinus_ROI.BWMask == 0, float('nan'), Sinus_ROI.BWMask)
        HbO2_output.Sinus_mean_timeseries = np.nanmean(
            Sinus_ROI_tmp * (Img_data.Echo2_deck - Img_data.Echo1_deck), axis=(0, 1))
        
        # Calculate mean phase difference in tissue ROI
        Tissue_ROI_tmp = np.where(Tissue_ROI.BWMask == 0, float('nan'), Tissue_ROI.BWMask)
        HbO2_output.Tissue_mean_timeseries = np.nanmean(
            Tissue_ROI_tmp * (Img_data.Echo2_deck - Img_data.Echo1_deck), axis=(0, 1))
       
        # Get parameters from GUI input fields
        deltaTe = Img_data.DeltaTE
        HbO2_output.Hct = float(e_Hct.get())
        HbO2_output.SS_angle = float(e_Deg.get())
        
        # Apply SBO equation
        # Constants: γ=267.513, Δχ_do=0.27, B0=3T, Δχ_0=0.0296
        HbO2_output.HbO2 = 1 - (
            2 * (HbO2_output.Sinus_mean_timeseries - HbO2_output.Tissue_mean_timeseries) / 
            (267.513 * 4 * math.pi * 0.27 * 3 * 
             (np.cos(np.radians(HbO2_output.SS_angle))**2 - 1/3) * 
             HbO2_output.Hct * deltaTe)
        ) - 0.0296
        
        # Display results
        HbO2_output.HbO2_str = "%5.2f" % HbO2_output.HbO2.mean()
        HbO2_str.set(HbO2_output.HbO2_str)
        
        # Update HbO2 plot
        HbO2_line_plot.set_ydata(HbO2_output.HbO2)
        HbO2_line_plot.set_xdata(range(Img_data.ndix[2]))
        ax_HbO2.set_xlim([0, Img_data.ndix[2]])
        ax_HbO2.set_ylim([np.min(HbO2_output.HbO2) * 0.9, np.max(HbO2_output.HbO2) * 1.1])
        canvas_flow.draw()  

    def set_Hct(Hct_input):
        """Set hematocrit value."""
        HbO2_output.Hct = Hct_input

    def set_SS_angle(SS_angle_input):
        """Set vessel angle relative to B0."""
        HbO2_output.SS_angle = SS_angle_input

    def calc_SS_angle(coords_sag_input, coords_cor_input):
        """
        Calculate the vessel-to-B0 angle from user-selected points on 
        sagittal and coronal angiographic views.
        
        Parameters
        ----------
        coords_sag_input : list of tuples
            Two points selected on the sagittal view
        coords_cor_input : list of tuples
            Two points selected on the coronal view
        """
        # Compute vessel direction vector from sagittal view
        vector_x = (coords_sag_input[1][0] - coords_sag_input[0][0], 
                     coords_sag_input[0][1] - coords_sag_input[1][1])
        vector_x_norm = vector_x / np.linalg.norm(vector_x)
        if np.angle(complex(vector_x_norm[0], vector_x_norm[1])) < 0:
            vector_x_norm = -1 * vector_x_norm
            
        # Compute vessel direction vector from coronal view
        vector_y = (-1 * coords_cor_input[1][0] + coords_cor_input[0][0], 
                     -1 * coords_cor_input[0][1] + coords_cor_input[1][1])
        vector_y_norm = vector_y / np.linalg.norm(vector_y)
        if np.angle(complex(vector_y_norm[0], vector_y_norm[1])) < 0:
            vector_y_norm = -1 * vector_y_norm
        
        # Combine into 3D vessel orientation vector
        vct = np.insert(vector_x_norm, 0, 0, axis=0) + np.array([vector_y_norm[0], 0, vector_y_norm[1]])
        vct_B0 = np.array([0, 0, 1])  # B0 field direction
    
        # Calculate angle between vessel vector and B0
        angle2 = math.atan2(np.linalg.norm(np.cross(vct, vct_B0)), np.dot(vct, vct_B0))
        HbO2_output.degree = 180 * angle2 / math.pi
        
        # Update vessel angle input field
        e_Deg.delete(0, 'end')
        e_Deg.insert(END, '{0:.1f}'.format(HbO2_output.degree))
        

# =============================================================================
# GUI BUTTONS - HbO2 CALCULATION
# =============================================================================
calc_button_group = LabelFrame(window, text='Calculate HbO2', borderwidth=2, relief='solid')
calc_button_group.grid(row=2, rowspan=1, column=1, columnspan=1, sticky='nw', padx=10, pady=40)

CalcHbO2_button = Button(master=calc_button_group, height=2, width=10,
                         text="Calculate HbO2", command=HbO2_output.Calc_HbO2)
CalcHbO2_button.grid(row=0, rowspan=2, column=0, columnspan=1, sticky='nw', pady=2, padx=10)

HbO2_text = Label(calc_button_group, text="Mean HbO2:")
HbO2_text.grid(row=2, rowspan=1, column=0, columnspan=1, sticky='nw')

HbO2_str = StringVar()
HbO2_text_str = Label(calc_button_group, textvariable=HbO2_str)
HbO2_text_str.grid(row=2, rowspan=1, column=0, columnspan=1, sticky='ne')

# Hematocrit input
Hct_text = Label(calc_button_group, text="Hematocrit:")
Hct_text.grid(row=0, rowspan=1, column=1, columnspan=1, sticky='nw', pady=2, padx=10)

e_Hct = Entry(calc_button_group, width=5)
e_Hct.grid(row=0, rowspan=1, column=2, columnspan=1, sticky='ne', pady=0, padx=0)
if args.hct_input is None:
    hct_str = '0.43'
else:
    hct_str = str(args.hct_input)
e_Hct.insert(END, hct_str)

# Vessel angle input
Deg_text = Label(calc_button_group, text="Vessel angle:")
Deg_text.grid(row=1, rowspan=1, column=1, columnspan=1, sticky='nw', pady=2, padx=10)

e_Deg = Entry(calc_button_group, width=5)
e_Deg.grid(row=1, rowspan=1, column=2, columnspan=1, sticky='ne', pady=0, padx=0)

if args.ss_angle_input is None:
    angle_ss_str = '0'
else:
    angle_ss_str = str(args.ss_angle_input)
e_Deg.insert(END, angle_ss_str)


# =============================================================================
# DATA EXPORT
# =============================================================================

data_saved_str = StringVar()

class save_ouput_data:
    """
    Class to handle saving analysis results to files.
    
    Exports HbO2 data to CSV and optionally saves:
    - ROI masks as NPZ (numpy archive) for Sinus and Tissue ROIs
    - ROI masks as NIfTI (labelled: Sinus=1, Tissue=2)
    - Animated GIF of ROI across frames
    
    Class Methods
    -------------
    save_data()
        Save all selected output files
    
    Output CSV Columns
    ------------------
    - HbO2: Oxygen saturation per frame
    - Sinus_TS: Mean inter-echo phase difference in sinus ROI per frame
    - Tissue_TS: Mean inter-echo phase difference in tissue ROI per frame
    - Hct: Hematocrit value
    - SS_angle: Vessel angle relative to B0 (degrees)
    """
    
    def save_data(self=''):
        """
        Save HbO2 data and optional ROI files.
        
        Reads filename from entry field and saves:
        - CSV with HbO2, phase differences, hematocrit, and vessel angle
        - NPZ with ROI polygons and masks (if checkbox selected)
        - NIfTI labelled ROI mask (if checkbox selected)        - GIF animation (if checkbox selected)
        """
        output_file = entry_save_filename.get()
        ouput_data_list = {
            'HbO2': HbO2_output.HbO2.tolist(), 
            'Sinus_TS': HbO2_output.Sinus_mean_timeseries.tolist(), 
            'Tissue_TS': HbO2_output.Tissue_mean_timeseries.tolist(),
            'Hct': [HbO2_output.Hct] * Img_data.ndix[2],
            'SS_angle': [HbO2_output.SS_angle] * Img_data.ndix[2]
        }
        df = pd.DataFrame(data=ouput_data_list)
        df.to_csv(output_file)
        data_saved_str.set('Data saved: ' + output_file)
        print('Data saved: ' + output_file)

        # Save ROI data as NPZ (numpy archive)
        if npz_roi.status:
            ROI_filename = os.path.splitext(output_file)[0] + '_ROIs'
            np.savez(ROI_filename, 
                     SinusROI_poly=np.array(Sinus_ROI.polygon, dtype='object'), 
                     SinusROI_BWMask=Sinus_ROI.BWMask, 
                     SinusROI_flag=Sinus_ROI.flag,
                     TissueROI_poly=np.array(Tissue_ROI.polygon, dtype='object'), 
                     TissueROI_BWMask=Tissue_ROI.BWMask, 
                     TissueROI_flag=Tissue_ROI.flag)

        # Save animated GIF of ROI overlays across frames
        if gif_roi.status:
            gif_dir = os.path.splitext(output_file)[0] + '_ROIgif'
            gif_basename = os.path.splitext((output_file.replace('/', ' ').split(' ')[-1]))[0]
            if not os.path.isdir(gif_dir): 
                os.mkdir(gif_dir)
            # Save HbO2 plot
            fig_flow.savefig(os.path.join(gif_dir, gif_basename + '_HbO2.png')) 
            # Save each frame with ROI overlay
            for i in range(1, Img_data.Echo1_deck.shape[2] + 1): 
                update_image(i) 
                fig.savefig(os.path.join(gif_dir, gif_basename + '_frame' + str(i) + '.png')) 
            create_gif(os.path.join(gif_dir, gif_basename)) 
            # Restore original frame
            update_image(imgFrame)
            slider_scale.set(imgFrame) 

        # Save ROI masks as NIfTI file
        # Creates a labelled mask: Sinus=1, Tissue=2
        # Works for both PAR/REC and NIfTI input data
        if nii_roi.status:
            ROI_combined = (Sinus_ROI.BWMask.astype(np.float32) * 1 + 
                            Tissue_ROI.BWMask.astype(np.float32) * 2)
            raw_img = nib.load(Img_data.raw_img_filename)
            ROI_nii = nib.Nifti1Image(
                np.expand_dims(np.flipud(np.rot90(ROI_combined)), 2), 
                affine=raw_img.affine
            ) 
            nib.save(ROI_nii, os.path.splitext(output_file)[0] + '_ROIs.nii')


def create_gif(path):
    """
    Create an animated GIF from saved frame images.
    
    Parameters
    ----------
    path : str
        Base path for frame images (without _frameN.png suffix)
    
    Notes
    -----
    Expects PNG files named {path}_frame1.png, {path}_frame2.png, etc.
    """
    frames = []
    imgs = glob.glob(path + "_frame*.png")
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(path + '.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=100, loop=0)
    return


class roi_save:
    """
    Helper class to track checkbox state for save options.
    
    Used for the NPZ, NIfTI, and GIF save checkboxes.
    
    Attributes
    ----------
    status : bool
        Current checkbox state (True = checked)
    
    Methods
    -------
    change_status()
        Toggle the status between True and False
    """
    def __init__(self):
        self.status = False
    def change_status(self):
        if self.status == True:
            self.status = False
        else:
            self.status = True

npz_roi = roi_save()
npz_roi.change_status()  # Default: NPZ saving enabled
gif_roi = roi_save()     # Default: GIF saving disabled
nii_roi = roi_save()
nii_roi.change_status()  # Default: NIfTI saving enabled


# =============================================================================
# GUI BUTTONS - SAVE DATA
# =============================================================================
Save_button_group = LabelFrame(window, text='Save data', borderwidth=2, relief='solid')
Save_button_group.grid(row=2, rowspan=1, column=1, columnspan=6, sticky='nw', padx=10, pady=140)

Save_button = Button(master=Save_button_group, height=2, width=8,
                     text="Save data", command=save_ouput_data.save_data)
Save_button.grid(row=0, rowspan=1, column=0, columnspan=1, sticky='sw', pady=2, padx=10)

# Checkboxes for optional save formats
Save_button.Save_check_roi_npz = Checkbutton(Save_button_group, text='.npz',
                                              variable=1, onvalue=1, offvalue=0, 
                                              command=npz_roi.change_status)
Save_button.Save_check_roi_npz.grid(row=0, rowspan=1, column=2, columnspan=1, sticky='sw', pady=2, padx=0)
Save_button.Save_check_roi_npz.select()

Save_button.Save_check_roi_nii = Checkbutton(Save_button_group, text='.nii',
                                              variable=2, onvalue=1, offvalue=0, 
                                              command=nii_roi.change_status)
Save_button.Save_check_roi_nii.grid(row=0, rowspan=1, column=3, columnspan=1, sticky='sw', pady=2, padx=0)
Save_button.Save_check_roi_nii.select()

Save_check_roi_gif = Checkbutton(Save_button_group, text='.gif',
                                  variable=3, onvalue=1, offvalue=0, 
                                  command=gif_roi.change_status)
Save_check_roi_gif.grid(row=0, rowspan=1, column=4, columnspan=1, sticky='sw', pady=2, padx=0)

# Output filename entry
save_str = StringVar()
entry_save_filename = Entry(Save_button_group, width=80, textvariable=save_str)
entry_save_filename.grid(row=3, rowspan=1, column=0, columnspan=12, sticky='nw', pady=0, padx=0)
save_str.set(Img_data.raw_img_filename[0:-4] + '_HbO2_data.csv')

Data_saved_txt = Label(Save_button_group, textvariable=data_saved_str)
Data_saved_txt.grid(row=2, rowspan=1, column=0, columnspan=12, sticky='se', pady=0, padx=0)


# =============================================================================
# VESSEL ANGLE ESTIMATION MODULE
# =============================================================================

def Calc_SS_angle():
    """
    Open the vessel-to-B0 angle estimation window.
    
    Loads angiographic MRI data and displays coronal and sagittal reformatted 
    views. The SBO acquisition plane is overlaid on the angiographic images 
    using the header information (off-centre positions and angulation). The 
    user identifies the vessel direction by clicking two points along the 
    vessel in each view, from which the 3D vessel orientation and angle 
    relative to B0 are computed.
    """
    Calc_SS_window = Tk()
    GUI_width = window.winfo_screenwidth() * 0.65
    GUI_height = window.winfo_screenheight() * 0.5
    Calc_SS_window.geometry(str(int(GUI_width)) + 'x' + str(int(GUI_height)))
    Calc_SS_window.resizable(True, True)
    Calc_SS_window.wm_title("Calculate vessel B0 angle")

    # Load angiographic data
    if args.angio_input is None:
        print('Select angiographic file')
        Angio_raw_header_filename = filedialog.askopenfilename(
            initialdir="/", title="Select file",
            filetypes=(("PAR files", "*.PAR"), ("all files", "*.*"))
        )        
    else:
        Angio_raw_header_filename = args.angio_input
    
    # Parse angiographic header
    head_obj_angio = open(Angio_raw_header_filename) 
    angio_header, angio_header_slice = nib.parrec.parse_PAR_header(head_obj_angio)
    angio_data = np.fromfile(Angio_raw_header_filename[0:-4] + '.REC', dtype='<f4')
    angio_dim = angio_header_slice['recon resolution'][0][0]
    angio_data_resh = angio_data[0:angio_dim * angio_dim].reshape(angio_dim, angio_dim)

    # Extract coronal and sagittal views from angiographic data
    angio_data_resh_coronal = (angio_data_resh[0:int(angio_dim / 2), 0:int(angio_dim / 2)] / 
                               angio_data_resh[0:int(angio_dim / 2), 0:int(angio_dim / 2)].max())
    angio_data_resh_sag = (angio_data_resh[int(angio_dim / 2):angio_dim, int(angio_dim / 2):angio_dim] / 
                           angio_data_resh[int(angio_dim / 2):angio_dim, int(angio_dim / 2):angio_dim].max())
    
    # --- Sagittal view ---
    fig_angio = plt.figure(figsize=(2.88, 2.88), dpi=100)
    ax_sag = fig_angio.add_subplot(111, position=[0, 0, 1, 1])
    angio_plot = ax_sag.imshow(angio_data_resh_coronal, vmin=0, vmax=0.0007, cmap='gray')
    ax_sag.axis('off')
    Sag_line_plot = ax_sag.plot([], [], '.r-')
    
    # Calculate and overlay the SBO acquisition plane on sagittal view
    pixel_size = angio_header['fov'][0] / angio_header['scan_resolution'][0]
    offcenter = angio_header['off_center'] - Img_data.pold_header['off_center']
    offcenter_n = offcenter / pixel_size
    
    v = np.array([0, 0, 1])  # Magnetic field vector 
    omega = (angio_header['angulation'][0] - Img_data.pold_header['angulation'][0])
    theta = -(angio_header['angulation'][2] - Img_data.pold_header['angulation'][2])

    # Rotation matrices for plane normal vector
    rot_mat_a = np.array([
        [1, 0, 0],
        [0, np.cos(np.radians(omega)), np.sin(np.radians(omega))],
        [0, -1 * np.sin(np.radians(omega)), np.cos(np.radians(omega))]
    ])
    rot_mat_b = np.array([
        [np.cos(np.radians(theta)), 0, -1 * np.sin(np.radians(theta))],
        [0, 1, 0],
        [np.sin(np.radians(theta)), 0, np.cos(np.radians(theta))]
    ])
    
    v_rot = np.matmul(v, np.matmul(rot_mat_a, rot_mat_b))
    Z = ((-1 * v_rot[1] * (np.array([-1 * angio_dim / 2, angio_dim / 4]) + offcenter_n[0])) / v_rot[2]) - offcenter_n[1]
    Sag_line_pold_plot = ax_sag.plot(
        [1, -1 + angio_dim / 2], 
        [angio_dim / 4 - Z[0], angio_dim / 4 - Z[1]], 
        'y-', linewidth=5, alpha=0.7
    )

    canvas_angio = FigureCanvasTkAgg(fig_angio, master=Calc_SS_window)
    canvas_angio.draw()
    canvas_angio.get_tk_widget().grid(row=0, rowspan=1, column=0, columnspan=1, padx=20, pady=20)
    ToolbarFrame_angio = Frame(Calc_SS_window)
    ToolbarFrame_angio.grid(row=1, column=0, rowspan=1, columnspan=1, padx=0, pady=0, sticky='nw')
    toobar_angio = NavigationToolbar2Tk(canvas_angio, ToolbarFrame_angio)

    # Sagittal view click handler for vessel direction identification
    global coords_sag
    coords_sag = []
    
    def onclick_sag(event):
        """Record click points on sagittal view for vessel direction."""
        coords_sag.append((event.xdata, event.ydata))
        Cor_line_plot_test = ax_sag.plot(coords_sag[0][0], coords_sag[0][1], '.r')
        canvas_angio.draw()

        if len(coords_sag) == 2:
            fig_angio.canvas.mpl_disconnect(cid)
            Cor_line_plot_test = ax_sag.plot(
                (coords_sag[0][0], coords_sag[1][0]), 
                (coords_sag[0][1], coords_sag[1][1]), '.r-'
            )    
            canvas_angio.draw()
     
    cid = fig_angio.canvas.mpl_connect('button_press_event', onclick_sag)
 
    # --- Coronal view ---
    fig_angio_cor = plt.figure(figsize=(2.88, 2.88), dpi=100)
    ax_cor = fig_angio_cor.add_subplot(111, position=[0, 0, 1, 1])
    angio_cor_plot = ax_cor.imshow(angio_data_resh_sag, vmin=0, vmax=0.0007, cmap='gray')
    ax_cor.axis('off')
    
    # Overlay SBO acquisition plane on coronal view
    Zy = ((v_rot[0] * (np.array([-1 * angio_dim / 2, angio_dim / 4]) + offcenter_n[2])) / v_rot[2]) - offcenter_n[1]
    Cor_line_pold_plot = ax_cor.plot(
        [1, -1 + angio_dim / 2], 
        [angio_dim / 4 - Zy[0], angio_dim / 4 - Zy[1]], 
        'y-', linewidth=5, alpha=0.7
    )

    canvas_angio_cor = FigureCanvasTkAgg(fig_angio_cor, master=Calc_SS_window)
    canvas_angio_cor.draw()
    canvas_angio_cor.get_tk_widget().grid(row=0, rowspan=1, column=1, columnspan=1, padx=20, pady=20)
    
    ToolbarFrame_angio = Frame(Calc_SS_window)
    ToolbarFrame_angio.grid(row=1, column=1, rowspan=1, columnspan=1, padx=0, pady=0, sticky='nw')
    toobar_angio = NavigationToolbar2Tk(canvas_angio_cor, ToolbarFrame_angio)
    
    # Coronal view click handler for vessel direction identification
    global coords_cor
    coords_cor = []
    
    def onclick_cor(event):
        """Record click points on coronal view for vessel direction."""
        coords_cor.append((event.xdata, event.ydata))
        Cor_line_plot_test = ax_cor.plot(coords_cor[0][0], coords_cor[0][1], '.r')
        canvas_angio_cor.draw()

        if len(coords_cor) == 2:
            fig_angio_cor.canvas.mpl_disconnect(cid)
            Cor_line_plot_test = ax_cor.plot(
                (coords_cor[0][0], coords_cor[1][0]), 
                (coords_cor[0][1], coords_cor[1][1]), '.r-'
            )    
            canvas_angio_cor.draw()
            # Calculate vessel angle from both views
            HbO2_output.calc_SS_angle(coords_sag, coords_cor)
            
    cid = fig_angio_cor.canvas.mpl_connect('button_press_event', onclick_cor)


# =============================================================================
# VESSEL ANGLE MENU
# =============================================================================
SSanglemenu = Menu(menubar, tearoff=1)
SSangle_submenu = Menu(SSanglemenu, tearoff=1)
menubar.add_cascade(label="Calc vessel B0 angle", menu=SSanglemenu)
SSanglemenu.add_command(label="Calc angle", command=Calc_SS_angle, accelerator="Control+o")


# =============================================================================
# ABOUT / HELP DIALOG
# =============================================================================

def popup_help():
    """Display the About dialog with software information."""
    popup = Tk()
    GUI_width = window.winfo_screenwidth() * 0.35
    GUI_height = window.winfo_screenheight() * 0.25
    popup.geometry(str(int(GUI_width)) + 'x' + str(int(GUI_height)))
    popup.resizable(True, True)
    popup.wm_title("About SBOCalculator")
    
    help_str = ('SBOCalculator: GUI for calculating blood oxygen saturation (HbO2) '
                'from susceptibility-based oximetry MRI data.\n'
                'Supports PAR/REC (Philips) and NIfTI file formats.')
    name_str = ('Mark B. Vestergaard\n'
                'mark.bitsch.vestergaard@regionh.dk\n'
                'Functional Imaging Unit\n'
                'Department of Clinical Physiology, Nuclear Medicine and PET\n'
                'Rigshospitalet, Glostrup, Denmark')
    
    text_title = Label(popup, text=help_str, anchor="w", background='white')
    text_title.pack(side="top", fill="x", pady=10)
    text_name = Label(popup, text=name_str, justify="left", anchor="w")
    text_name.pack(side="top", fill="x", pady=10)
    
    B1 = Button(popup, text="Close", command=popup.destroy)
    B1.pack(side="top", pady=10)
    popup.mainloop()

def load_ROI_file(self=''):
    """
    Load previously saved ROI data from NPZ file.
    
    Opens file dialog to select an NPZ file containing:
    - SinusROI_poly: Sinus polygon vertices per frame
    - SinusROI_BWMask: Sinus binary mask array
    - SinusROI_flag: Sinus ROI status flags
    - TissueROI_poly: Tissue polygon vertices per frame
    - TissueROI_BWMask: Tissue binary mask array
    - TissueROI_flag: Tissue ROI status flags
    
    Keyboard shortcut: Ctrl+R
    """
    ROI_filename = filedialog.askopenfilename(
        initialdir="/", title="Select file",
        filetypes=(("ROI files", "*.npz"), ("all files", "*.*"))
    )
    Loaded_ROI = np.load(ROI_filename, allow_pickle=True)
    Sinus_ROI.polygon = Loaded_ROI['SinusROI_poly'].tolist()
    Sinus_ROI.BWMask = Loaded_ROI['SinusROI_BWMask']
    Sinus_ROI.flag = Loaded_ROI['SinusROI_flag'].tolist()
    Tissue_ROI.polygon = Loaded_ROI['TissueROI_poly'].tolist()
    Tissue_ROI.BWMask = Loaded_ROI['TissueROI_BWMask']
    Tissue_ROI.flag = Loaded_ROI['TissueROI_flag'].tolist()
    global imgFrame
    update_image(imgFrame)

# Load ROI shortcut
analysismenu.add_command(label="Load ROI file", command=load_ROI_file, accelerator="Control+r")
window.bind_all('<Control-Key-r>', func=load_ROI_file)

# Save data shortcut
analysismenu.add_command(label="Save data", command=save_ouput_data.save_data, accelerator="Control+s")
window.bind_all('<Control-Key-s>', func=save_ouput_data.save_data)

analysismenu.add_command(label="About", command=popup_help, accelerator="Control+a")
window.bind_all('<Control-Key-a>', func=popup_help)


# =============================================================================
# LAUNCH APPLICATION
# =============================================================================
window.config(menu=menubar)
window.mainloop()
