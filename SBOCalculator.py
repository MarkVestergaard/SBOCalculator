#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# SBOCalculator

GUI for analyzing susceptibility-based oximetry MRI data for measuring blood saturation (Jain et al. Journal of Cerebral Blood Flow & Metabolism 2010, Vol.30 (9) 1598–1607, https://doi.org/10.1038/jcbfm.2010.49) 

Written for python 3.8. Tested on SBO data from 3T Philips dSTREAM Achieva MRI and 3T Siemens Biograph mMR hybrid PET/MR system.

Input for Philips scanner data: **.PAR file
Input for Siemens scanner data: nifti-file converted from dicom using dcmniix (https://github.com/rordenlab/dcm2niix)

Region of interest (ROI) is manual delineated. Data saved as csv file.

For in-depth description of analysis see: Vestergaard et al. Cerebral Cortex, Volume 32, Issue 6, 15 March 2022, 1295–1306, doi:https://doi.org/10.1093/cercor/bhab294 or Vestergaard et al. Journal of Cerebral Blood Flow & Metabolism 2019, Vol. 39(5) 834–848, doi:https://doi.org/10.1177/0271678X17737909

Mark B. Vestergaard
Functional Imaging Unit,
Department of Clinical Physiology and Nuclear Medicine
Rigshospitalet Copenhagen, Denmark
mark.bitsch.vestergaard@regionh.dk

@author: Mark B. Vestegaard, June 2021, mark.bitsch.vestergaard@regionh.dk 
"""

from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import nibabel as nib
# OBS: To Load POLD data line "466: _chk_trunc('dynamic scan', 'max_dynamics')" in nibabel.parrec needs to be omitted. 
import argparse
import numpy as np
import scipy
import numpy.matlib
import pandas as pd
#from functools import partial    
from matplotlib.widgets import PolygonSelector
from matplotlib.patches import Polygon 
from matplotlib.path import Path
from skimage import measure
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw
import os 


#import csv
import os 
import math
#import scipy.io

# parser for loading PAR/REC file
parser = argparse.ArgumentParser(description='Calculate flow in vessel')
parser.add_argument('--img', dest='img_input', help='Input PCM image')
parser.add_argument('--angio', dest='angio_input', help='Input angio image')
parser.add_argument('--hct', dest='hct_input', help='Input Hematocrite')
parser.add_argument('--angle', dest='ss_angle_input', help='Input vessel angle')
args = parser.parse_args()


# The main tkinter GUI window
window = Tk()
window.title('Calculate SBO')
GUI_width = window.winfo_screenwidth()*0.8
GUI_height = window.winfo_screenheight()*1
window.geometry( str(int(GUI_width)) +'x'+ str(int(GUI_height)) )
window.resizable(1,1)

global nifti_files
nifti_files=None

# Load PCM data
if args.img_input==None: # Open dialog if no argument
    print('Select SBO file')
    raw_img_filename = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("Philips PAR","*.PAR"),("NIFTI","*.nii"),("all files","*.*")),multiple=True)        
    args.img_input=raw_img_filename
    file_type = os.path.splitext(raw_img_filename[0])[1]
    if file_type=='.nii':
        nifti_files=True
        import json
else: # Set argument as filename
    raw_img_filename = args.img_input # 


    

# Class of PCM image data
class Img_data_parrec():
    def __init__(self,raw_img_filename):
        self.hdr_obj=open(raw_img_filename) 
        self.pold_header, self.pold_header_slice_info =nib.parrec.parse_PAR_header(self.hdr_obj)
        self.raw_img = nib.load(raw_img_filename)# Loads PCM par file nibabel load
        self.raw_img_filename=raw_img_filename
        self.img = nib.as_closest_canonical(self.raw_img) # Loads PCM par file
        self.nifti_file = nib.Nifti1Image(self.img.dataobj, self.img.affine, header=self.img.header)
        self.Venc=self.img.header.general_info.get('phase_enc_velocity')[2] # Find venc in header
    # Split image
    
        self.Phase_deck_tmp=np.rot90(np.arctan2(self.img.dataobj[:,:,0,self.pold_header_slice_info['image_type_mr']==2],self.img.dataobj[:,:,0,self.pold_header_slice_info['image_type_mr']==1]))
        self.Mod_deck_tmp=np.rot90(self.img.dataobj[:,:,0,np.logical_and(self.pold_header_slice_info['image_type_mr']==0, self.pold_header_slice_info['echo number']==1)])
        self.Mod_deck=((self.Mod_deck_tmp/self.Mod_deck_tmp.max())*2*math.pi)-math.pi
        self.vel_indx=np.sort(list(range(0,self.Phase_deck_tmp.shape[2]-1,4))+list(range(1,self.Phase_deck_tmp.shape[2]-1,4))) # Removes vel encoded frames
        self.Phase_deck=np.delete(self.Phase_deck_tmp,self.vel_indx,2)
        self.Echo1_deck=self.Phase_deck[:,:,range(0,self.Phase_deck.shape[2],2)]
        self.Echo2_deck=self.Phase_deck[:,:,range(1,self.Phase_deck.shape[2],2)]
        self.Echo_diff_deck=self.Echo2_deck-self.Echo1_deck
        self.ndix=self.Echo1_deck.shape
        self.DeltaTE=(self.pold_header_slice_info['echo_time'].max()-self.pold_header_slice_info['echo_time'].min())/1000
  
    
    def set_new_data(self): # Set new data
        self.raw_img_filename = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("PAR files","*.PAR"),("all files","*.*")))        
        print(self.raw_img_filename)
        print(self.hdr_obj)
        self.hdr_obj=open(self.raw_img_filename) 
        self.pold_header, self.pold_header_slice_info =nib.parrec.parse_PAR_header(self.hdr_obj)
        self.raw_img = nib.load(self.raw_img_filename)# Loads PCM par file nibabel load
        self.raw_img_filename=self.raw_img_filename
        self.img = nib.as_closest_canonical(self.raw_img) # Loads PCM par file
        self.nifti_file = nib.Nifti1Image(self.img.dataobj, self.img.affine, header=self.img.header)
        self.Venc=self.img.header.general_info.get('phase_enc_velocity')[2] # Find venc in header
    # Split image
    
        self.Phase_deck_tmp=np.rot90(np.arctan2(self.img.dataobj[:,:,0,self.pold_header_slice_info['image_type_mr']==2],self.img.dataobj[:,:,0,self.pold_header_slice_info['image_type_mr']==1]))
        self.Mod_deck_tmp=np.rot90(self.img.dataobj[:,:,0,np.logical_and(self.pold_header_slice_info['image_type_mr']==0, self.pold_header_slice_info['echo number']==1)])
        self.Mod_deck=((self.Mod_deck_tmp/self.Mod_deck_tmp.max())*2*math.pi)-math.pi
        self.vel_indx=np.sort(list(range(0,self.Phase_deck_tmp.shape[2]-1,4))+list(range(1,self.Phase_deck_tmp.shape[2]-1,4))) # Removes vel encoded frames
        self.Phase_deck=np.delete(self.Phase_deck_tmp,self.vel_indx,2)
        self.Echo1_deck=self.Phase_deck[:,:,range(0,self.Phase_deck.shape[2],2)]
        self.Echo2_deck=self.Phase_deck[:,:,range(1,self.Phase_deck.shape[2],2)]
        self.Echo_diff_deck=self.Echo2_deck-self.Echo1_deck
        self.ndix=self.Echo1_deck.shape    
        self.DeltaTE=(self.pold_header_slice_info['echo_time'].max()-self.pold_header_slice_info['echo_time'].min())/1000
  
           
        global Disp_image_str
        global colormap_str
        change_image(Disp_image_str,colormap_str)
        save_str.set(Img_data.raw_img_filename[0:-4]+'_HbO2_data.csv')


# Class of PCM image data
class Img_data_nii():
    def __init__(self,raw_img_filename):
        self.raw_img = nib.load(raw_img_filename[0])# Loads PCM par file nibabel load
        self.raw_img_filename=raw_img_filename[0]
        self.img = nib.as_closest_canonical(self.raw_img) # Loads PCM par file
        self.nifti_file = nib.Nifti1Image(self.img.dataobj, self.img.affine, header=self.img.header)
        fname_json_tmp = open(os.path.splitext(raw_img_filename[0])[0]+'.json')
        self.img.json_header = json.load(fname_json_tmp)     
        ImageType_a=self.img.json_header['ImageType'][2]
        echo_number_a=self.img.json_header['EchoNumber']
        
        self.raw_img_b = nib.load(raw_img_filename[1])# Loads PCM par file nibabel load
        self.raw_img_filename_b=raw_img_filename[1]
        self.img_b = nib.as_closest_canonical(self.raw_img_b) # Loads PCM par file
        self.nifti_file_b = nib.Nifti1Image(self.img_b.dataobj, self.img_b.affine, header=self.img_b.header)
        fname_json_tmp = open(os.path.splitext(raw_img_filename[1])[0]+'.json')
        self.img_b.json_header = json.load(fname_json_tmp)     
        ImageType_b=self.img_b.json_header['ImageType'][2]
        echo_number_b=self.img_b.json_header['EchoNumber']
        
        self.raw_img_c = nib.load(raw_img_filename[2])# Loads PCM par file nibabel load
        self.raw_img_filename_b=raw_img_filename[2]
        self.img_c = nib.as_closest_canonical(self.raw_img_c) # Loads PCM par file
        self.nifti_file_c = nib.Nifti1Image(self.img_c.dataobj, self.img_c.affine, header=self.img_c.header)
        fname_json_tmp = open(os.path.splitext(raw_img_filename[2])[0]+'.json')
        self.img_c.json_header = json.load(fname_json_tmp)     
        ImageType_c=self.img_c.json_header['ImageType'][2]
        echo_number_c=self.img_c.json_header['EchoNumber']
        
        self.raw_img_d = nib.load(raw_img_filename[3])# Loads PCM par file nibabel load
        self.raw_img_filename_d=raw_img_filename[3]
        self.img_d = nib.as_closest_canonical(self.raw_img_d) # Loads PCM par file
        self.nifti_file_d = nib.Nifti1Image(self.img_d.dataobj, self.img_d.affine, header=self.img_d.header)
        fname_json_tmp = open(os.path.splitext(raw_img_filename[3])[0]+'.json')
        self.img_d.json_header = json.load(fname_json_tmp)     
        ImageType_d=self.img_d.json_header['ImageType'][2]
        echo_number_d=self.img_d.json_header['EchoNumber']

        self.ImageType=[ImageType_a, ImageType_b, ImageType_c, ImageType_d]
        self.EchoNumber=[echo_number_a, echo_number_b, echo_number_c, echo_number_d]
        
        indx_M=[index for index, value in enumerate(self.ImageType) if value == 'M']
        indx_P=[index for index, value in enumerate(self.ImageType) if value == 'P']
        indx_ec1=[index for index, value in enumerate(self.EchoNumber) if value == 1]
        indx_ec2=[index for index, value in enumerate(self.EchoNumber) if value == 2]
        
        
        def common(a,b): 
            c = [value for value in a if value in b] 
            return c
        M_ec1_ind=common(indx_M,indx_ec1)[0];
        P_ec1_ind=common(indx_P,indx_ec1)[0];
        P_ec2_ind=common(indx_P,indx_ec2)[0];
        
        
        if self.EchoNumber.index(1)==0:
            self.Echotime1=self.img.json_header['EchoTime']
        if self.EchoNumber.index(1)==1:
            self.Echotime1=self.img_b.json_header['EchoTime']
        if self.EchoNumber.index(1)==2:
            self.Echotime1=self.img_c.json_header['EchoTime']
            
        if self.EchoNumber.index(2)==0:
            self.Echotime2=self.img.json_header['EchoTime']
        if self.EchoNumber.index(2)==1:
            self.Echotime2=self.img_b.json_header['EchoTime']
        if self.EchoNumber.index(2)==2:
            self.Echotime2=self.img_c.json_header['EchoTime']
        
        self.DeltaTE=self.Echotime2-self.Echotime1


        if M_ec1_ind==0:
            self.Mod_deck=np.rot90(  self.nifti_file.dataobj/self.nifti_file.dataobj.max()   )*np.pi
        elif M_ec1_ind==1:
            self.Mod_deck=np.rot90(  self.nifti_file_b.dataobj/self.nifti_file_b.dataobj.max()  )*np.pi
        elif M_ec1_ind==2:
            self.Mod_deck=np.rot90(  self.nifti_file_c.dataobj/self.nifti_file_c.dataobj.max()   )*np.pi
        elif M_ec1_ind==3:
            self.Mod_deck=np.rot90(   self.nifti_file_d.dataobj/self.nifti_file_d.dataobj.max()   )*np.pi
 
        if P_ec1_ind==0:
            self.Echo1_deck=np.rot90( self.nifti_file.dataobj )/2048
        elif P_ec1_ind==1:
            self.Echo1_deck=np.rot90(  self.nifti_file_b.dataobj  )/2048
        elif P_ec1_ind==2:
            self.Echo1_deck=np.rot90( self.nifti_file_c.dataobj )/2048
        elif P_ec1_ind==3:
            self.Echo1_deck=np.rot90(  self.nifti_file_d.dataobj  )/2048
            
        if P_ec2_ind==0:
            self.Echo2_deck=np.rot90( self.nifti_file.dataobj  )/2048
        elif P_ec2_ind==1:
            self.Echo2_deck=np.rot90( self.nifti_file_b.dataobj  )/2048
        elif P_ec2_ind==2:
            self.Echo2_deck=np.rot90( self.nifti_file_c.dataobj   )/2048
        elif P_ec2_ind==3:
            self.Echo2_deck=np.rot90( self.nifti_file_d.dataobj  )/2048
         

        self.Echo_diff_deck=self.Echo2_deck-self.Echo1_deck
        self.ndix=self.Echo1_deck.shape

    def set_new_data(self):
        print('Test')



if not nifti_files:
    Img_data=Img_data_parrec(raw_img_filename[0])        

if nifti_files:
    Img_data=Img_data_nii(raw_img_filename)  
        
        
   

# Function for specifying image data type
global Disp_image_str
Disp_image_str='1echo'
global colormap_str
colormap_str='jet'
global imgFrame
imgFrame=1

def displayed_image(disp_name): # Returns Disp_image variable with shown data
    if disp_name == '1echo':
        return Img_data.Echo1_deck
    elif disp_name == '2echo':
        return Img_data.Echo2_deck
    elif disp_name == 'mod':
        return Img_data.Mod_deck
    elif disp_name == 'Echo_diff':
        return Img_data.Echo2_deck-Img_data.Echo1_deck
    elif disp_name == 'ROI':
        return Sinus_ROI.BWMask*Img_data.Venc*0.5
Disp_image=displayed_image(Disp_image_str)

# Function for changing image
def change_image(image_str,cmp_str): # Changed displayed image
    Disp_image=displayed_image(image_str)
    POLD_plot.set_data(Disp_image[:,:,int(imgFrame-1)])
    POLD_plot.set_cmap(cmp_str)
    POLD_plot.set_clim(climits.lims[0],climits.lims[1])
    canvas.draw()
    return Disp_image_str


# Function for capturing change of image from topmenu or keyboard (could be changed to a class)
def change_image_type_str_1echo(self=''):
    global Disp_image_str
    Disp_image_str='1echo'
    change_image(Disp_image_str,colormap_str)
    
def change_image_type_str_2echo(self=''):
    global Disp_image_str
    Disp_image_str='2echo'
    change_image(Disp_image_str,colormap_str)

def change_image_type_str_mod(self=''):
    global Disp_image_str
    Disp_image_str='mod'
    change_image(Disp_image_str,colormap_str)

def change_image_type_str_echo_diff(self=''):
    global Disp_image_str
    Disp_image_str='Echo_diff'
    change_image(Disp_image_str,colormap_str)
    
def change_image_type_str_ROI(self=''):
    global Disp_image_str
    Disp_image_str='ROI'
    change_image(Disp_image_str,colormap_str)




# Function for capturing change of colorbar from topmenu or keyboard (could be changed to a class)
def change_cmap_jet(self='jet'):
    global colormap_str
    colormap_str='jet'
    change_image(Disp_image_str,colormap_str)

def change_cmap_gray(self='gray'):
    global colormap_str
    colormap_str='gray'
    change_image(Disp_image_str,colormap_str)

def change_cmap_viridis(self='viridis'):
    global colormap_str
    colormap_str='viridis'
    change_image(Disp_image_str,colormap_str)

    

def popup_change_colorbar():
    popup_change_colorbar = Tk()
    popup_change_colorbar.title('Colorbar limits')
    GUI_width = window.winfo_screenwidth()*0.10
    GUI_height = window.winfo_screenheight()*0.22
    popup_change_colorbar.geometry( str(int(GUI_width)) +'x'+ str(int(GUI_height)) )
    popup_change_colorbar.resizable(True,True)
    
    #SinusROI_button_group = LabelFrame(window, text='Sinus ROI', borderwidth=2,relief='solid')
    #SinusROI_button_group.grid(row=1,rowspan=1,column=0,columnspan=1,sticky='nw',padx=10,pady=20)

    Min_entry_text = Label(popup_change_colorbar, text="Min:",width = 7,padx=0)
    Min_entry_text.grid(row=1,rowspan=1,column=0,columnspan=1,sticky='n',pady=10,padx=0)
    Min_entry = Entry(popup_change_colorbar,width=7)
    Min_entry.grid(row=1,rowspan=1,column=1,columnspan=1,sticky='nw',pady=10,padx=0)
    Min_entry.insert(END, '-3.140')
    
    Max_entry_text = Label(popup_change_colorbar, text="Max:",width = 7,padx=0)
    Max_entry_text.grid(row=0,rowspan=1,column=0,columnspan=1,sticky='n',pady=0,padx=0)
    Max_entry = Entry(popup_change_colorbar,width=7)
    Max_entry.grid(row=0,rowspan=1,column=1,columnspan=1,sticky='nw',pady=0,padx=0)
    Max_entry.insert(END, '3.140')
    
    def update_colorbar():
        climits.lims=[float(Min_entry.get()),float(Max_entry.get())]
        print('test')
        change_image(Disp_image_str,colormap_str)
        
    
    UpdateCB_button = Button(master = popup_change_colorbar,
                      height = 2,
                      width = 9,
                     text = "Update Colorbar", command=update_colorbar)
    UpdateCB_button.grid(row=2,rowspan=1,column=0,columnspan=2,sticky='nw',pady=2,padx=10)
    
    
    

## Create topmenu 
menubar = Menu(window)
window.config(menu=menubar)


# New file menu
analysismenu = Menu(menubar, tearoff=1)
analysis_submenu = Menu(analysismenu, tearoff=1)
menubar.add_cascade(label="File", menu=analysismenu)
analysismenu.add_command(label="Load new file", command=Img_data.set_new_data,accelerator="Control+n")
window.bind_all('<Control-Key-n>', func=Img_data.set_new_data )


# Image type topmenu
imagemenu = Menu(menubar, tearoff=1)
submenu = Menu(imagemenu, tearoff=1)
menubar.add_cascade(label="Image", menu=imagemenu)
imagemenu.add_cascade(label="Change image type", menu=submenu)

submenu.add_radiobutton(label="Echo 1",accelerator="Control+1",command = change_image_type_str_1echo)
submenu.add_radiobutton(label="Echo 2",accelerator="Control+2",command = change_image_type_str_2echo)
submenu.add_radiobutton(label="Echo 2 - Echo 1",accelerator="Control+3",command = change_image_type_str_echo_diff)
submenu.add_radiobutton(label="Modulus",accelerator="Control+4",command = change_image_type_str_mod)
submenu.add_radiobutton(label="ROI mask",accelerator="Control+5",command  = change_image_type_str_ROI)

window.bind_all('<Control-Key-1>', func=change_image_type_str_1echo ) # Binds keyboard shortcut to topbar 
window.bind_all('<Control-Key-2>', func=change_image_type_str_2echo )
window.bind_all('<Control-Key-3>', func=change_image_type_str_echo_diff )
window.bind_all('<Control-Key-4>', func=change_image_type_str_mod )
window.bind_all('<Control-Key-5>', func=change_image_type_str_ROI )


# Colormap topmenu
subsubmenu = Menu(submenu, tearoff=1)
imagemenu.add_cascade(label="Change colorbar", menu=subsubmenu)
subsubmenu.add_command(label="Jet",accelerator="Control+q", command = change_cmap_jet)
subsubmenu.add_command(label="Gray",accelerator="Control+w", command = change_cmap_gray)
subsubmenu.add_command(label="Viridis",accelerator="Control+e" ,command = change_cmap_viridis)
subsubmenu.add_command(label="Change colorbar limit",accelerator="Control+l" ,command = popup_change_colorbar)


window.bind_all('<Control-Key-q>', func=change_cmap_jet ) # Binds keyboard shortcut to topbar 
window.bind_all('<Control-Key-w>', func=change_cmap_gray )
window.bind_all('<Control-Key-e>', func=change_cmap_viridis )


# Create grid for GUI
#Grid.rowconfigure(window, 0, weight=0)
#Grid.columnconfigure(window, 0, weight=0)

#Grid.rowconfigure(window,0,weight=1)
Grid.columnconfigure(window,1,weight=1)
Grid.columnconfigure(window,0,weight=1)
Grid.columnconfigure(window,5,weight=1)
#Grid.rowconfigure(window,1,weight=1)
#Grid.columnconfigure(window,1,weight=1)
#Grid.rowconfigure(window,2,weight=1)
#Grid.columnconfigure(window,2,weight=1)

# Create figure frame for displaying MRI image
fig = plt.figure(figsize=(5.1, 5.1), dpi=100)
ax = fig.add_subplot(111)

class climits():
    lims=[-3.14,3.14]
    
POLD_plot=ax.imshow(Disp_image[:,:,imgFrame-1],cmap=plt.get_cmap(colormap_str),vmin=climits.lims[0],vmax=climits.lims[1], interpolation='none')
fig.colorbar(POLD_plot, ax=ax, shrink=0.6)


ax.set_xticks([]) # Removes x axis ticks
ax.set_yticks([]) # Removes x axis ticks
fig.tight_layout() # Tight layout

Sinus_ROI_line_plot, = ax.plot([], [], '.w-')
Tissue_ROI_line_plot, = ax.plot([], [], '.b-')
Unwrap_ROI_line_plot, = ax.plot([], [], '.c-')

canvas = FigureCanvasTkAgg(fig, master=window)  # Draws the figure in the tkinter GUI window
canvas.draw()
canvas.get_tk_widget().grid(row=1,rowspan=1,column=1,columnspan=1,padx=0,pady=0) # place in grid

# Create line data for plotting ROI data
fig_flow = plt.figure(figsize=(3,3), dpi=110)
ax_HbO2 = fig_flow.add_subplot(111)


# Create line plot for flow data
ax_HbO2.set_position([0.2, 0.15, 0.7, 0.7])
ax_HbO2.tick_params(labelsize=7)
ax_HbO2.set_ylabel('HbO2', fontsize = 8.0) # Y label
ax_HbO2.set_xlabel('Index', fontsize = 8.0) # Y label
HbO2_line_plot, = ax_HbO2.plot([], [], '.r-')

canvas_flow = FigureCanvasTkAgg(fig_flow, master=window)  # Draws the figure in the tkinter GUI window
canvas_flow.draw()
canvas_flow.get_tk_widget().grid(row=1,rowspan=1,column=5,columnspan=1,padx=20,pady=50) # place in grid


# Create frame for the navigation toolbar
ToolbarFrame = Frame(window)
ToolbarFrame.grid(row=2, column=1,rowspan=1,columnspan=1,padx=0,pady=0,sticky='nw')
toobar = NavigationToolbar2Tk(canvas, ToolbarFrame)


# Function for changing the image data and ROI data
def update_image(self):
    global imgFrame
    imgFrame=int(self)
    Disp_image=displayed_image(Disp_image_str)
    POLD_plot.set_data(Disp_image[:,:,int(imgFrame-1)])
    #POLD_plot.set_clim(0,10)
    Sinus_ROI_as_array = np.array(Sinus_ROI.polygon[int(imgFrame-1)])
    if Sinus_ROI_as_array.any() == None:
        Sinus_ROI_line_plot.set_ydata([])
        Sinus_ROI_line_plot.set_xdata([])
    else:
        ROI_as_array_tmp=np.append(Sinus_ROI_as_array[:,:], Sinus_ROI_as_array[0,:]).reshape(Sinus_ROI_as_array.shape[0]+1,Sinus_ROI_as_array.shape[1])
        Sinus_ROI_line_plot.set_ydata(ROI_as_array_tmp[:,1])
        Sinus_ROI_line_plot.set_xdata(ROI_as_array_tmp[:,0])
    Tissue_ROI_as_array = np.array(Tissue_ROI.polygon[int(imgFrame-1)])
    if Tissue_ROI_as_array.any() == None:
        Tissue_ROI_line_plot.set_ydata([])
        Tissue_ROI_line_plot.set_xdata([])
    else:
        Tissue_ROI_as_array_tmp=np.append(Tissue_ROI_as_array[:,:], Tissue_ROI_as_array[0,:]).reshape(Tissue_ROI_as_array.shape[0]+1,Tissue_ROI_as_array.shape[1])
        Tissue_ROI_line_plot.set_ydata(Tissue_ROI_as_array_tmp[:,1])
        Tissue_ROI_line_plot.set_xdata(Tissue_ROI_as_array_tmp[:,0])
        
        
    canvas.draw()    
    return imgFrame


# Create slider for changing frame.
slider_scale = Scale(window, to=1, from_=Img_data.Echo1_deck.shape[2], width=20,length=400 ,command=update_image)
slider_scale.grid(row=1,rowspan=1,column=4,columnspan=1,padx=0,pady=0,sticky='w')


def key_arrow_up(self=''):
    global imgFrame
    if imgFrame>(Img_data.Echo1_deck.shape[2]-1):
        imgFrame=imgFrame
    else:
        imgFrame=imgFrame+1
    update_image(imgFrame)
    slider_scale.set(imgFrame)
    print(imgFrame)

window.bind_all('<Up>', func=key_arrow_up )       
    
def key_arrow_down(self=''):
    global imgFrame
    if imgFrame<2:
        imgFrame=imgFrame
    else:
        imgFrame=imgFrame-1
    update_image(imgFrame)
    slider_scale.set(imgFrame)
    print(imgFrame)

window.bind_all('<Down>', func=key_arrow_down )    





# Add headline text
headline_text = Label(window, text="Draw ROI to calculate oxygen saturation the blood vessel")
headline_text.grid(row=0,rowspan=1,column=0,columnspan=2,sticky='nw')


# Class of ROI data
class ROI:
    def __init__(self):
        self.polygon=[None] * Img_data.Echo1_deck.shape[2]
        self.BWMask=Img_data.Echo1_deck*False
        self.flag=[0]*(Img_data.ndix[2])

    def set_polygon(self, verts, curr_frame):
        print('Updating ROI')
        self.polygon[int(curr_frame-1)]=verts # ROI as polygon as input
        self.flag[int(curr_frame-1)]=1
        
        path = Path(verts)
        x, y = np.meshgrid(np.arange(Img_data.ndix[0]), np.arange(Img_data.ndix[1]))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x,y)).T
        grid = path.contains_points(points) # Find voxels inside polygon
        grid = grid.reshape((Img_data.ndix[0],Img_data.ndix[1]))        
        self.BWMask[:,:,int(curr_frame-1)]=grid
    def loaded_roi_from_file(Loaded_ROI_data):
        self.polygon=Loaded_ROI_data['PCMROI_poly'].tolist()
        self.BWMask=Loaded_ROI_data['PCMROI_BWMask']
        self.flag=Loaded_ROI_data['PCMROI_flag']
        global imgFrame
        print(imgFrame)
        update_image(imgFrame)
        
Sinus_ROI=ROI()
Tissue_ROI=ROI()
Unwrap_ROI=ROI()
# Class for drawing ROI as a polygon



class ROIPolygon(object):
    def __init__(self,name, ax, row, col):
        self.name=name
        self.canvas = ax.figure.canvas
        self.polygon=''
        self.PS = PolygonSelector(ax,
                                    self.onselect,
                                    lineprops = dict(color = 'k', alpha = 0.5),
                                    markerprops = dict(mec = 'k', mfc = 'k', alpha = 0.5, markersize=3),vertex_select_radius=10)
        self.PS.set_visible(False)
        self.PS.set_active(False)
        #ROIPolygon.poly=PS 
    def onselect(self, verts):
        #print(verts) 
        self.canvas.draw_idle()
        self.polygons=verts
        
        if self.name=='SS':
            Sinus_ROI.set_polygon(verts,imgFrame)
            Sinus_ROI.flag[imgFrame-1]=1
            Sinus_ROI_as_array = np.array(self.PS.verts)
            Sinus_ROI_as_array_tmp=np.append(Sinus_ROI_as_array[:,:], Sinus_ROI_as_array[0,:]).reshape(Sinus_ROI_as_array.shape[0]+1,Sinus_ROI_as_array.shape[1])
            Sinus_ROI_line_plot.set_ydata(Sinus_ROI_as_array_tmp[:,1])
            Sinus_ROI_line_plot.set_xdata(Sinus_ROI_as_array_tmp[:,0])
            global Disp_image_str
            global colormap_str
            change_image(Disp_image_str,colormap_str)
    

            
        if self.name=='Tissue':
            Tissue_ROI.set_polygon(verts,imgFrame)
            Tissue_ROI.flag[imgFrame-1]=1
            Tissue_ROI_as_array = np.array(self.PS.verts)
            Tissue_ROI_as_array_tmp=np.append(Tissue_ROI_as_array[:,:], Tissue_ROI_as_array[0,:]).reshape(Tissue_ROI_as_array.shape[0]+1,Tissue_ROI_as_array.shape[1])
            Tissue_ROI_line_plot.set_ydata(Tissue_ROI_as_array_tmp[:,1])
            Tissue_ROI_line_plot.set_xdata(Tissue_ROI_as_array_tmp[:,0])
      #      global Disp_image_str
      #      global colormap_str
            change_image(Disp_image_str,colormap_str)
         #   self.PS.set_visible(False)
         #   self.PS.set_active(False)
        if self.name=='Unwrap':
            Unwrap_ROI.set_polygon(verts,imgFrame)
            Unwrap_ROI.flag[imgFrame-1]=1
            Unwrap_ROI_as_array = np.array(self.PS.verts)
            Unwrap_ROI_as_array_tmp=np.append(Unwrap_ROI_as_array[:,:], Unwrap_ROI_as_array[0,:]).reshape(Unwrap_ROI_as_array.shape[0]+1,Unwrap_ROI_as_array.shape[1])
            Unwrap_ROI_line_plot.set_ydata(Unwrap_ROI_as_array_tmp[:,1])
            Unwrap_ROI_line_plot.set_xdata(Unwrap_ROI_as_array_tmp[:,0])
      #      global Disp_image_str
      #      global colormap_str
            change_image(Disp_image_str,colormap_str)
         #   self.PS.set_visible(False)
         #   self.PS.set_active(False)
        self.canvas.draw_idle()
      #  self.PS.set_visible(False)
      #  self.PS.set_active(False)                             

#PS_SS=ROIPolygon('SS', POLD_plot.axes, Img_data.ndix[0], Img_data.ndix[1])
#PS_tissue=ROIPolygon('Tissue',POLD_plot.axes, Img_data.ndix[0], Img_data.ndix[1])
#PS_unwrap=ROIPolygon('Unwrap',POLD_plot.axes, Img_data.ndix[0], Img_data.ndix[1])
        
  #      ROI_as_array = np.array(PS.verts)
  #      ROI_as_array_tmp=np.append(ROI_as_array[:,:], ROI_as_array[0,:]).reshape(ROI_as_array.shape[0]+1,ROI_as_array.shape[1])
  #      ROI_line_plot.set_ydata(ROI_as_array_tmp[:,1])
  #      ROI_line_plot.set_xdata(ROI_as_array_tmp[:,0])
  #      global Disp_image_str
  #      global colormap_str
  #      change_image(Disp_image_str,colormap_str)
        
      #  ROI_as_array = np.array(PS.verts)
      #  ROI_as_array_tmp=np.append(ROI_as_array[:,:], ROI_as_array[0,:]).reshape(ROI_as_array.shape[0]+1,ROI_as_array.shape[1])
      #  ROI_line_plot.set_ydata(ROI_as_array_tmp[:,1])
      #  ROI_line_plot.set_xdata(ROI_as_array_tmp[:,0])
      #  global Disp_image_str
      #  global colormap_str
      #  change_image(Disp_image_str,colormap_str)

    
 # Function for initiating sinus ROI polygon 
class AddSinusROI():
    def add_roi():
        if hasattr(AddSinusROI, 'PS_SS'):
            AddSinusROI.PS_SS.PS.set_visible(False)
            AddSinusROI.PS_SS.PS.set_active(False)
        AddSinusROI.PS_SS=ROIPolygon('SS', POLD_plot.axes, Img_data.ndix[0], Img_data.ndix[1])
        AddSinusROI.PS_SS.PS.set_visible(True)
        AddSinusROI.PS_SS.PS.set_active(True)
        UpdateSinusROI_button.configure(text='Stop ROI edit')
        canvas.draw() 

def CopySinusROI():
    first_indx=Sinus_ROI.flag.index(1)
    if Sinus_ROI.flag[int(imgFrame-1)]==1:
        for x in range(len(Sinus_ROI.polygon)):
            Sinus_ROI.polygon[x]=Sinus_ROI.polygon[int(imgFrame-1)]
            Sinus_ROI.BWMask[:,:,x]=Sinus_ROI.BWMask[:,:,int(imgFrame-1)]
            Sinus_ROI.flag[x]=1
    else:
        for x in range(len(Sinus_ROI.polygon)): 
            Sinus_ROI.polygon[x]=Sinus_ROI.polygon[first_indx]
            Sinus_ROI.BWMask[:,:,x]=Sinus_ROI.BWMask[:,:,first_indx]
            Sinus_ROI.flag[x]=1

def UpdateSinusROI():
    if  AddSinusROI.PS_SS.PS.get_active():
        #print('Remove ROI')
        AddSinusROI.PS_SS.PS.set_visible(False)
        AddSinusROI.PS_SS.PS.set_active(False)
        canvas.draw() 
        UpdateSinusROI_button.configure(text='Edit ROI')
    else:
        print('Updating ROI')

        AddSinusROI.PS_SS.PS.set_visible(True)
        AddSinusROI.PS_SS.PS.set_active(True)
        canvas.draw() 
        UpdateSinusROI_button.configure(text='Stop ROI edit')

 # Function for initiating Tissue ROI polygon 
class AddTissueROI():
    def add_roi():
        if hasattr(AddTissueROI, 'PS_tissue'):
            AddTissueROI.PS_tissue.PS.set_visible(False)
            AddTissueROI.PS_tissue.PS.set_active(False)
        AddTissueROI.PS_tissue=ROIPolygon('Tissue', POLD_plot.axes, Img_data.ndix[0], Img_data.ndix[1])
        AddTissueROI.PS_tissue.PS.set_visible(True)
        AddTissueROI.PS_tissue.PS.set_active(True)
        canvas.draw() 
        UpdateTissueROI_button.configure(text='Stop ROI edit')

def CopyTissueROI():
    first_indx=Tissue_ROI.flag.index(1)
    if Tissue_ROI.flag[int(imgFrame-1)]==1:
        for x in range(len(Tissue_ROI.polygon)):
            Tissue_ROI.polygon[x]=Tissue_ROI.polygon[int(imgFrame-1)]
            Tissue_ROI.BWMask[:,:,x]=Tissue_ROI.BWMask[:,:,int(imgFrame-1)]
            Tissue_ROI.flag[x]=1
    else:
        for x in range(len(Tissue_ROI.polygon)): 
            Tissue_ROI.polygon[x]=Tissue_ROI.polygon[first_indx]
            Tissue_ROI.BWMask[:,:,x]=Tissue_ROI.BWMask[:,:,first_indx]
            Tissue_ROI.flag[x]=1

def UpdateTissueROI():
    if  AddTissueROI.PS_tissue.PS.get_active():
        #print('Remove ROI')
        AddTissueROI.PS_tissue.PS.set_visible(False)
        AddTissueROI.PS_tissue.PS.set_active(False)
        canvas.draw() 
        UpdateTissueROI_button.configure(text='Edit ROI')
    else:
        print('Updating ROI')

        AddTissueROI.PS_tissue.PS.set_visible(True)
        AddTissueROI.PS_tissue.PS.set_active(True)
        canvas.draw() 
        UpdateTissueROI_button.configure(text='Stop ROI edit')

 # Function for initiating Uwnrap ROI polygon 
class AddUnwrapROI():
    def add_roi():
        if hasattr(AddUnwrapROI, 'PS_tissue'):
            AddUnwrapROI.PS_unwrap.PS.set_visible(False)
            AddUnwrapROI.PS_unwrap.PS.set_active(False)
        AddUnwrapROI.PS_unwrap=ROIPolygon('Unwrap', POLD_plot.axes, Img_data.ndix[0], Img_data.ndix[1])
        AddUnwrapROI.PS_unwrap.PS.set_visible(True)
        AddUnwrapROI.PS_unwrap.PS.set_active(True)
        canvas.draw() 
        UpdateUnwrapROI_button.configure(text='Stop ROI edit')

def CopyUnwrapROI():
    first_indx=Unwrap_ROI.flag.index(1)
    if Unwrap_ROI.flag[int(imgFrame-1)]==1:
        for x in range(len(Unwrap_ROI.polygon)):
            Unwrap_ROI.polygon[x]=Unwrap_ROI.polygon[int(imgFrame-1)]
            Unwrap_ROI.BWMask[:,:,x]=Unwrap_ROI.BWMask[:,:,int(imgFrame-1)]
            Unwrap_ROI.flag[x]=1
    else:
        for x in range(len(Unwrap_ROI.polygon)): 
            Unwrap_ROI.polygon[x]=Unwrap_ROI.polygon[first_indx]
            Unwrap_ROI.BWMask[:,:,x]=Unwrap_ROI.BWMask[:,:,first_indx]
            Unwrap_ROI.flag[x]=1

def UpdateUnwrapROI():
    if  AddUnwrapROI.PS_unwrap.PS.get_active():
        #print('Remove ROI')
        AddUnwrapROI.PS_unwrap.PS.set_visible(False)
        AddUnwrapROI.PS_unwrap.PS.set_active(False)
        canvas.draw() 
        UpdateUnwrapROI_button.configure(text='ROI edit')
    else:
        print('Updating ROI')

        AddUnwrapROI.PS_unwrap.PS.set_visible(True)
        AddUnwrapROI.PS_unwrap.PS.set_active(True)
        canvas.draw()
        UpdateUnwrapROI_button.configure(text='Stop ROI edit')

def RemoveUnwrapROI():
    Unwrap_ROI=ROI()
    Unwrap_ROI_line_plot.set_ydata([None])
    Unwrap_ROI_line_plot.set_xdata([None])
    change_image(Disp_image_str,colormap_str)
    


# Function for unwrapping images 

def UnwrapImage():
    Unwrap_ROI.BWMask=np.where(Unwrap_ROI.BWMask==0, float('nan'), Unwrap_ROI.BWMask)
    if Disp_image_str == '1echo':
            EchoWunrap_tmp=Unwrap_ROI.BWMask*Img_data.Echo1_deck
            Voxels_for_unwrap_indx=EchoWunrap_tmp<float(e_unwrap.get())
            #EchoWunrap_tmp[Voxels_for_unwrap_indx]=EchoWunrap_tmp[Voxels_for_unwrap_indx]+math.pi*2
            Img_data.Echo1_deck[Voxels_for_unwrap_indx]=EchoWunrap_tmp[Voxels_for_unwrap_indx]+math.pi*2
            Img_data.Echo_diff_deck=Img_data.Echo2_deck-Img_data.Echo1_deck
    if Disp_image_str == '2echo':
            EchoWunrap_tmp=Unwrap_ROI.BWMask*Img_data.Echo2_deck
            Voxels_for_unwrap_indx=EchoWunrap_tmp<float(e_unwrap.get())
            #EchoWunrap_tmp[Voxels_for_unwrap_indx]=EchoWunrap_tmp[Voxels_for_unwrap_indx]+math.pi*2
            Img_data.Echo2_deck[Voxels_for_unwrap_indx]=EchoWunrap_tmp[Voxels_for_unwrap_indx]+math.pi*2
            Img_data.Echo_diff_deck=Img_data.Echo2_deck-Img_data.Echo1_deck
    change_image(Disp_image_str,colormap_str)
            
        
            




# # Function for automatic delineation of ROI by region growing algorithm
class RegGrow():

     nRow, nCol, nSlice=Img_data.Echo1_deck.shape
     qu=1
     ginput_input=[]
     btm_press_event=[]
    
     def create_mask(event):
         window.config(cursor='')
         ginput_input= event.xdata, event.ydata
         maxDist=int(e1.get())
         ThresVal=float(e2.get()) 
         p=0
         fig.canvas.callbacks.disconnect(RegGrow.btm_press_event)
         seed_tmp=np.round(ginput_input)
         seed=np.flip(seed_tmp)
         Reg_mask=np.zeros(Img_data.ndix)
         Reg_mask_tmp=np.zeros(Img_data.ndix)

         
         if Disp_image_str=='1echo':
             Thres_image_tmp=Img_data.Echo1_deck
        
         if Disp_image_str=='2echo':
             Thres_image_tmp=Img_data.Echo2_deck
         
         if Disp_image_str=='Echo_diff':
             Thres_image_tmp=Img_data.Echo_diff_deck
             
         if Disp_image_str=='mod':
             Thres_image_tmp=Img_data.Mod_deck   

  
         for nn in range(Img_data.ndix[2]):
             queue = seed
             Imax=int(seed[0])
             Jmax=int(seed[1])
             while queue.any():
                 #print(queue)
                 if queue.ndim==1:
                     xv = int(queue[0])
                     yv = int(queue[1])
                 else:
                     xv = int(queue[0][0])
                     yv = int(queue[0][1])
                    
                 for n in [-1,0,1]:
                     # print(n)
                     for m in [-1,0,1]:
                         #print(m)
                         if xv+n > 0  and  xv+n <= RegGrow.nRow and yv+m > 0 and  yv+m <= RegGrow.nCol and any([n, m]) and Reg_mask_tmp[xv+n,yv+m,nn]==0 and np.sqrt( (xv+n-Imax)**2 + (yv+m-Jmax)**2 ) < maxDist and Thres_image_tmp[xv+n,yv+m,nn] >= ThresVal:
                             Reg_mask_tmp[(xv+n, yv+m,nn)]=1
                             queue=np.vstack((queue, np.array([xv+n, yv+m])))
                         #print(queue)   
                 print(nn)
                 Reg_mask[:,:,nn]=scipy.ndimage.binary_erosion(Reg_mask_tmp[:,:,nn],iterations=2).astype(np.float32)
                 queue = numpy.delete(queue, (0), axis=0)
                 RegGrow.qu=queue  
                 masked_inp_tmp=Img_data.Echo_diff_deck[:,:,nn]*Reg_mask[:,:,nn]
                 New_seed = np.where(masked_inp_tmp == np.amax(masked_inp_tmp))
                 seed=np.array( [New_seed[0][0], New_seed[1][0]])
             print(nn)
         for mm in range(Img_data.ndix[2]):
             found_contours=measure.find_contours(Reg_mask[:,:,mm], 0.5)
             cnt_length=np.zeros(len(found_contours))
             for bb in range(len(found_contours)):
                     cnt_length[bb]=found_contours[bb].shape[0]
             max_count_idx=np.where(cnt_length == np.amax(cnt_length))   
             Sinus_ROI.set_polygon(np.fliplr(found_contours[max_count_idx[0][0]]),mm+1)
         update_image(imgFrame)

            

#         global PS 
#         ROIPolygon(pcm_plot.axes, Img_data.Vel_image.shape[0], Img_data.Vel_image.shape[2])
#         PS.set_visible(False)
#         PS.set_active(False) 
#         ROI_as_array = np.array(ROI_art.polygon[int(imgFrame-1)])
#         ROI_line_plot.set_ydata(ROI_as_array[:,1])
 #        ROI_line_plot.set_xdata(ROI_as_array[:,0])
#         fig.canvas.draw()
        

     def Calc_Auto_ROI() :
         window.config(cursor='plus green white')
         RegGrow.btm_press_event=fig.canvas.callbacks.connect('button_press_event', RegGrow.create_mask)

def donothing():
    print('Do nothing')
    
 # Create buttons for Sinus ROI controls 
SinusROI_button_group = LabelFrame(window, text='Sinus ROI', borderwidth=2,relief='solid')
SinusROI_button_group.grid(row=1,rowspan=1,column=0,columnspan=1,sticky='nw',padx=10,pady=20)

 # Add new ROI
AddSinusROI_button = Button(master = SinusROI_button_group,
                      height = 2,
                      width = 10,
                     text = "Add ROI", command=AddSinusROI.add_roi)
AddSinusROI_button.grid(row=0,rowspan=1,column=0,columnspan=2,sticky='nw',pady=2,padx=10)

 # Update ROI
UpdateSinusROI_button = Button(master = SinusROI_button_group,
                      height = 2,
                      width = 10,
                     text = "Edit ROI", command=UpdateSinusROI)
UpdateSinusROI_button.grid(row=1,rowspan=1,column=0,columnspan=2,sticky='nw',pady=2,padx=10)

 # Copy ROI to remaning frames 
CopySinusROI_button = Button(master = SinusROI_button_group,
                      height = 2,
                      width = 10,
                     text = "Copy ROI to \n all frames", command=CopySinusROI)
CopySinusROI_button.grid(row=3,rowspan=1,column=0,columnspan=2,sticky='nw',pady=2,padx=10)

 # Call region growing algorithm
AutoROI_button = Button(master = SinusROI_button_group,
                      height = 3,
                      width = 10,
                     text = "Automatic \n delineation", command=RegGrow.Calc_Auto_ROI)
AutoROI_button.grid(row=4,rowspan=1,column=0,columnspan=2,sticky='nw',pady=2,padx=10)
Max_dist_text = Label(SinusROI_button_group, text="Max. dist:",width = 7,padx=0)
Max_dist_text.grid(row=5,rowspan=1,column=0,columnspan=1,sticky='n',pady=0,padx=0)

 # Entries for region growing algorithm 
e1 = Entry(SinusROI_button_group,width=4)
e1.grid(row=5,rowspan=1,column=1,columnspan=1,sticky='nw',pady=0,padx=0)
e1.insert(END, '7')

Threshold_text = Label(SinusROI_button_group, text="Threshold:")
Threshold_text.grid(row=6,rowspan=1,column=0,columnspan=1,sticky='nw',pady=0,padx=10)
e2 = Entry(SinusROI_button_group,width=4)
e2.grid(row=6,rowspan=1,column=1,columnspan=1,sticky='nw',pady=0,padx=0)
e2.insert(END, '1')


 # Create buttons for Tissue ROI controls 
TissueROI_button_group = LabelFrame(window, text='Tissue ROI', borderwidth=2,relief='solid')
TissueROI_button_group.grid(row=1,rowspan=1,column=0,columnspan=1,sticky='sw',padx=10,pady=20)

 # Add new ROI
AddTissueROI_button = Button(master = TissueROI_button_group,
                      height = 2,
                      width = 10,
                     text = "Add ROI", command=AddTissueROI.add_roi)
AddTissueROI_button.grid(row=0,rowspan=1,column=0,columnspan=2,sticky='nw',pady=2,padx=10)

 # Update ROI
UpdateTissueROI_button = Button(master = TissueROI_button_group,
                      height = 2,
                      width = 10,
                     text = "Edit ROI", command=UpdateTissueROI)
UpdateTissueROI_button.grid(row=1,rowspan=1,column=0,columnspan=2,sticky='nw',pady=2,padx=10)

 # Copy ROI to remaning frames 
CopyTissueROI_button = Button(master = TissueROI_button_group,
                      height = 2,
                      width = 10,
                     text = "Copy ROI to \n all frames", command=CopyTissueROI)
CopyTissueROI_button.grid(row=3,rowspan=1,column=0,columnspan=2,sticky='nw',pady=2,padx=10)

Unwrap_button_group = LabelFrame(window, text='Unwrap image', borderwidth=2,relief='solid')
Unwrap_button_group.grid(row=2,rowspan=1,column=0,columnspan=1,sticky='nw',padx=10,pady=1)

 # Add new ROI
AddUnwrapROI_button = Button(master = Unwrap_button_group,
                      height = 2,
                      width = 10,
                     text = "Add ROI", command=AddUnwrapROI.add_roi)
AddUnwrapROI_button.grid(row=0,rowspan=1,column=0,columnspan=2,sticky='nw',pady=2,padx=10)
 
 # Update ROI
UpdateUnwrapROI_button = Button(master = Unwrap_button_group,
                      height = 2,
                      width = 10,
                     text = "Edit ROI", command=UpdateUnwrapROI)
UpdateUnwrapROI_button.grid(row=1,rowspan=1,column=0,columnspan=2,sticky='nw',pady=2,padx=10)

 # Copy ROI to remaning frames 
CopyUnwrapROI_button = Button(master = Unwrap_button_group,
                      height = 2,
                      width = 10,
                     text = "Copy ROI to \n all frames", command=CopyUnwrapROI)
CopyUnwrapROI_button.grid(row=2,rowspan=1,column=0,columnspan=2,sticky='nw',pady=2,padx=10)

RemoveUnwrapROI_button = Button(master = Unwrap_button_group,
                      height = 2,
                      width = 10,
                     text = "Remove ROI", command=RemoveUnwrapROI)
RemoveUnwrapROI_button.grid(row=3,rowspan=1,column=0,columnspan=2,sticky='nw',pady=2,padx=10)

UnwrapLimit = Label(Unwrap_button_group, text="Unwrap limit:",width = 9,padx=0)
UnwrapLimit.grid(row=4,rowspan=1,column=0,columnspan=1,sticky='nw',pady=0,padx=0)

 # Entries for region growing algorithm 
e_unwrap = Entry(Unwrap_button_group,width=4)
e_unwrap.grid(row=4,rowspan=1,column=1,columnspan=1,sticky='ne',pady=0,padx=0)
e_unwrap.insert(END, '0')

Unwrap_button = Button(master = Unwrap_button_group,
                      height = 2,
                      width = 10,
                     text = "Unwrap", command=UnwrapImage)
Unwrap_button.grid(row=5,rowspan=1,column=0,columnspan=2,sticky='nw',pady=2,padx=10)



class HbO2_output:
    def __init__():
        if args.hct_input==None:
            HbO2_output.Hct=0.43
        else: 
            HbO2_output=args.hct_input
        
        if args.ss_angle_input==None:
            HbO2_output.SS_angle=0
        else: 
            HbO2_output.SS_angle=args.ss_angle_input
        
        
        
    def Calc_HbO2():
        Sinus_ROI_tmp=np.where(Sinus_ROI.BWMask==0, float('nan'), Sinus_ROI.BWMask)
        HbO2_output.Sinus_mean_timeseries=np.nanmean( Sinus_ROI_tmp*(Img_data.Echo2_deck-Img_data.Echo1_deck), axis=(0,1))
        
        Tissue_ROI_tmp=np.where(Tissue_ROI.BWMask==0, float('nan'), Tissue_ROI.BWMask)
        HbO2_output.Tissue_mean_timeseries=np.nanmean( Tissue_ROI_tmp*(Img_data.Echo2_deck-Img_data.Echo1_deck), axis=(0,1))
       
        #deltaTe=(Img_data.pold_header_slice_info['echo_time'].max()-Img_data.pold_header_slice_info['echo_time'].min())/1000
        deltaTe=Img_data.DeltaTE
        HbO2_output.Hct=float(e_Hct.get())
        HbO2_output.SS_angle=float(e_Deg.get())
        
        HbO2_output.HbO2=1-(2*(HbO2_output.Sinus_mean_timeseries-(HbO2_output.Tissue_mean_timeseries))/(267.513*4*math.pi*0.27*3*(np.cos(np.radians(HbO2_output.SS_angle))-1/3)*HbO2_output.Hct*deltaTe))-0.0296;
        
        print('HbO2:')
        print(HbO2_output.HbO2)
        HbO2_output.HbO2_str="%5.2f" % HbO2_output.HbO2.mean()
        HbO2_str.set(HbO2_output.HbO2_str)
        
        HbO2_line_plot.set_ydata(HbO2_output.HbO2)
        HbO2_line_plot.set_xdata(range(Img_data.ndix[2]))
        ax_HbO2.set_xlim([0,Img_data.ndix[2]])
        ax_HbO2.set_ylim([np.min(HbO2_output.HbO2)*0.9,np.max(HbO2_output.HbO2)*1.1])
        canvas_flow.draw()  
    def set_Hct(Hct_input):
        HbO2_output.Hct=Hct_input
    def set_SS_angle(SS_angle_input):
        HbO2_output.SS_angle=SS_angle_input
    def calc_SS_angle(coords_sag_input,coords_cor_input):
        print('Calc SS angle')
        vector_x=(coords_sag_input[1][0]-coords_sag_input[0][0],coords_sag_input[0][1]-coords_sag_input[1][1])
        vector_x_norm=vector_x/np.linalg.norm(vector_x)
        if np.angle(complex(vector_x_norm[0],vector_x_norm[1])) < 0:
            vector_x_norm=-1*vector_x_norm
            
            
        vector_y=(-1*coords_cor_input[1][0]+coords_cor_input[0][0],-1*coords_cor_input[0][1]+coords_cor_input[1][1])
        vector_y_norm=vector_y/np.linalg.norm(vector_y)
        if np.angle(complex(vector_y_norm[0],vector_y_norm[1])) < 0:
            vector_y_norm=-1*vector_y_norm
        
        vct=np.insert(vector_x_norm, 0, 0, axis=0)+np.array([vector_y_norm[0],0,vector_y_norm[1]])
        vct_B0=np.array([0,0,1])
    
        
        angle2 = math.atan2(np.linalg.norm(np.cross(vct,vct_B0)), np.dot(vct,vct_B0))
        HbO2_output.degree=180*angle2/math.pi
        print(HbO2_output.degree)
        e_Deg.delete(0, 'end')
        e_Deg.insert(END, '{0:.1f}'.format(HbO2_output.degree))
        
# Button group for calculating flow
calc_button_group = LabelFrame(window, text='Calculate HbO2', borderwidth=2,relief='solid')
calc_button_group.grid(row=2,rowspan=1,column=1,columnspan=1,sticky='nw',padx=10,pady=40)
CalcHbO2_button = Button(master = calc_button_group,
                      height = 2,
                      width = 10,
                      text = "Calculate HbO2", command=HbO2_output.Calc_HbO2)
CalcHbO2_button.grid(row=0,rowspan=2,column=0,columnspan=1,sticky='nw',pady=2,padx=10)
HbO2_text = Label(calc_button_group, text="Mean HbO2:")
HbO2_text.grid(row=2,rowspan=1,column=0,columnspan=1,sticky='nw')
HbO2_str=StringVar()
HbO2_text_str = Label(calc_button_group,textvariable=HbO2_str)
HbO2_text_str.grid(row=2,rowspan=1,column=0,columnspan=1,sticky='ne')
Hct_text = Label(calc_button_group, text="Hematocrite:")
Hct_text.grid(row=0,rowspan=1,column=1,columnspan=1,sticky='nw',pady=2,padx=10)

e_Hct = Entry(calc_button_group,width=5)
e_Hct.grid(row=0,rowspan=1,column=2,columnspan=1,sticky='ne',pady=0,padx=0)
if args.hct_input == None:
    hct_str='0.43'
else:
    hct_str=str(args.hct_input)
e_Hct.insert(END, hct_str)


Deg_text = Label(calc_button_group, text="Vessel angle:")
Deg_text.grid(row=1,rowspan=1,column=1,columnspan=1,sticky='nw',pady=2,padx=10)

e_Deg = Entry(calc_button_group,width=5)
e_Deg.grid(row=1,rowspan=1,column=2,columnspan=1,sticky='ne',pady=0,padx=0)

if args.ss_angle_input == None:
    angle_ss_str='0'
else:
    angle_ss_str=str(args.ss_angle_input)

e_Deg.insert(END, angle_ss_str)



data_saved_str=StringVar()
class save_ouput_data:
    #output_file=args.img_input[0:-4]+'_Flow_data.csv'
    
    def save_data(self=''):
        print('test')
        output_file=entry_save_filename.get()
        ouput_data_list={'HbO2': HbO2_output.HbO2.tolist(), 'Sinus_TS': HbO2_output.Sinus_mean_timeseries.tolist(), 'Tissue_TS':HbO2_output.Tissue_mean_timeseries.tolist(),'Hct':[HbO2_output.Hct]*Img_data.ndix[2],'SS_angle':[HbO2_output.SS_angle]*Img_data.ndix[2]}
        df = pd.DataFrame(data=ouput_data_list)
        df.to_csv(output_file)
        data_saved_str.set('Data saved:'+output_file)
        print('Data saved:'+output_file)
        #ROI_filename=os.path.splitext(output_file)[0]+'_ROIs' # Also save ROI as npz data
        #np.savez(ROI_filename, PCMROI_poly=np.array(ROI_art.polygon,dtype='object'), PCMROI_BWMask=ROI_art.BWMask, PCMROI_flag=ROI_art.flag)



# Button groups for saving data
Save_button_group = LabelFrame(window, text='Save data', borderwidth=2,relief='solid')
Save_button_group.grid(row=2,rowspan=1,column=1,columnspan=6,sticky='nw',padx=10,pady=140)
Save_button = Button(master = Save_button_group,
                     height = 2,
                     width = 8,
                    text = "Save data", command=save_ouput_data.save_data)
Save_button.grid(row=0,rowspan=1,column=0,columnspan=1,sticky='sw',pady=2,padx=10)

# Entry for output file 
save_str=StringVar()
entry_save_filename=Entry(Save_button_group,width=80,textvariable=save_str)
entry_save_filename.grid(row=2,rowspan=1,column=0,columnspan=4,sticky='nw',pady=0,padx=0)
save_str.set(Img_data.raw_img_filename[0:-4]+'_SBO_data.csv')

Data_saved_txt = Label(Save_button_group, textvariable=data_saved_str)
Data_saved_txt.grid(row=1,rowspan=1,column=0,columnspan=1,sticky='se',pady=0,padx=0)





def Calc_SS_angle():
    Calc_SS_window = Tk()
    GUI_width = window.winfo_screenwidth()*0.65
    GUI_height = window.winfo_screenheight()*0.6
    Calc_SS_window.geometry( str(int(GUI_width)) +'x'+ str(int(GUI_height)) )
    Calc_SS_window.resizable(True,True)


    Calc_SS_window.wm_title("Calculate vessel BO angle")
    print('Select Angio file')
    
    if args.angio_input==None: # Open dialog if no argument
        print('Select PCM file')
        Angio_raw_header_filename = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("PAR files","*.PAR"),("all files","*.*")))        
        
    else: # Set argument as filename
        Angio_raw_header_filename = args.angio_input # 
    
    
    head_obj_angio=open(Angio_raw_header_filename) 
    angio_header, angio_header_slice =nib.parrec.parse_PAR_header(head_obj_angio)
    #args.img_input=raw_img_filename
    angio_data = np.fromfile(Angio_raw_header_filename[0:-4]+'.REC', dtype='<f4')
    angio_dim=angio_header_slice['recon resolution'][0][0];
    angio_data_resh=angio_data[0:angio_dim*angio_dim].reshape(angio_dim,angio_dim)

    angio_data_resh_coronal=angio_data_resh[0:int(angio_dim/2),0:int(angio_dim/2)]/angio_data_resh[0:int(angio_dim/2),0:int(angio_dim/2)].max()
    angio_data_resh_sag=angio_data_resh[int(angio_dim/2):angio_dim,int(angio_dim/2):angio_dim]/angio_data_resh[int(angio_dim/2):angio_dim,int(angio_dim/2):angio_dim].max()
    
    fig_angio = plt.figure(figsize=(2.88, 2.88), dpi=100)
    ax_sag = fig_angio.add_subplot(111,position=[0,0,1,1])
    angio_plot=ax_sag.imshow(angio_data_resh_coronal,vmin=0,vmax=0.0007,cmap='gray')
    Sag_line_plot = ax_sag.plot([], [], '.r-')
    

   
    pixel_size=angio_header['fov'][0]/angio_header['scan_resolution'][0]
    offcenter=angio_header['off_center']-Img_data.pold_header['off_center'] #Off Centre midslice(ap,fh,rl) [mm] 
    offcenter_n=offcenter/pixel_size; #Off Centre midslice(ap,fh,rl) [pixels] 
    
    v=np.array([0,0,1]); # Magnetic field vector 
    omega=(angio_header['angulation'][0]-Img_data.pold_header['angulation'][0] ) # angulation (ap,fh,rl) degree
    theta=-(angio_header['angulation'][2]-Img_data.pold_header['angulation'][2] ) # OBS: 3 vs 1. 

    # % Rotates planes normalvector to the offcenter and angulation
    rot_mat_a=np.array( [[1,0,0],[0, np.cos(np.radians(omega)), np.sin(np.radians(omega))],[0, -1*np.sin(np.radians(omega)), np.cos(np.radians(omega))] ] )
    rot_mat_b= np.array( [[np.cos(np.radians(theta)),0,-1*np.sin(np.radians(theta))], [0,1,0],[np.sin(np.radians(theta)),0,np.cos(np.radians(theta)) ]])
    
    v_rot=np.matmul(v,np.matmul(rot_mat_a,rot_mat_b))
   # center_point = ax_sag.plot(angio_dim/4-offcenter_n[0], angio_dim/4-offcenter_n[1], 'or-')
    Z = ((-1*v_rot[1]*(np.array([-1*angio_dim/2,angio_dim/4])+offcenter_n[0]))/ v_rot[2])-offcenter_n[1]
    Sag_line_pold_plot = ax_sag.plot([1,-1+angio_dim/2], [   angio_dim/4-Z[0] ,  angio_dim/4-Z[1] ], 'y-',linewidth=5,alpha=0.7)

    
    canvas_angio = FigureCanvasTkAgg(fig_angio, master=Calc_SS_window)  # Draws the figure in the tkinter GUI window
    canvas_angio.draw()
    canvas_angio.get_tk_widget().grid(row=0,rowspan=1,column=0,columnspan=1,padx=20,pady=20) # place in grid


    global coords_sag
    coords_sag=[]
    
    def onclick_sag(event):
        print(coords_sag)
    
        coords_sag.append((event.x, event.y))
        print(coords_sag)
        Sag_line_plot[0].set_xdata( np.array([coords_sag[0][0]]   ))
        Sag_line_plot[0].set_ydata( np.array([(angio_dim/2)-coords_sag[0][1]])   )
        canvas_angio.draw()

        if len(coords_sag) == 2:
            fig_angio.canvas.mpl_disconnect(cid)
            Sag_line_plot[0].set_xdata( np.array([coords_sag[0][0],coords_sag[1][0] ])   )
            Sag_line_plot[0].set_ydata( np.array([(angio_dim/2)-coords_sag[0][1],(angio_dim/2)-coords_sag[1][1]])   )
            #Sag_line_plot[0].set_xdata( np.array([30,50])   )
            #Sag_line_plot[0].set_ydata( np.array([100,120])   )
            canvas_angio.draw()
        
            
    cid = fig_angio.canvas.mpl_connect('button_press_event', onclick_sag)
    
    
    
    fig_angio_cor = plt.figure(figsize=(2.88, 2.88), dpi=100)
    ax_cor = fig_angio_cor.add_subplot(111,position=[0,0,1,1])
    angio_cor_plot=ax_cor.imshow(angio_data_resh_sag,vmin=0,vmax=0.0007,cmap='gray')
    #Cor_line_plot = ax_cor.plot([], [], '.r-')    
    
    
#    Zy= ((v_rot(1)*([-info.Dimensions(1)/2 info.Dimensions(1)/2]-offcenter_n(3)))./(v_rot(3)))-offcenter_n(2);

    Zy = ((v_rot[0]*(np.array([-1*angio_dim/2,angio_dim/4])+offcenter_n[2]))/ v_rot[2])-offcenter_n[1]
    Cor_line_pold_plot = ax_cor.plot([1,-1+angio_dim/2], [   angio_dim/4-Zy[0] ,  angio_dim/4-Zy[1] ], 'y-',linewidth=5,alpha=0.7)

    
    
    canvas_angio_cor = FigureCanvasTkAgg(fig_angio_cor, master=Calc_SS_window)  # Draws the figure in the tkinter GUI window
    canvas_angio_cor.draw()
    canvas_angio_cor.get_tk_widget().grid(row=0,rowspan=1,column=1,columnspan=1,padx=20,pady=20) # place in grid

    
    
    global coords_cor
    coords_cor=[]
    
    def onclick_cor(event):
        print(coords_cor)
    
        coords_cor.append((event.x, event.y))
        print(coords_cor)
        Cor_line_plot_test = ax_cor.plot(coords_cor[0][0],(angio_dim/2)-coords_cor[0][1],'.r')
        #Cor_line_plot_test[0].set_ydata( np.array([(angio_dim/2)-coords_cor[0][1]])   )
        canvas_angio_cor.draw()

        if len(coords_cor) == 2:
            fig_angio_cor.canvas.mpl_disconnect(cid)
            #Cor_line_plot[0].set_xdata( np.array([coords_cor[0][0],coords_cor[1][0] ])   )
            #Cor_line_plot[0].set_ydata( np.array([coords_cor[1][1],coords_cor[0][1] ])   )
            Cor_line_plot_test = ax_cor.plot((coords_cor[0][0],coords_cor[1][0]), ((angio_dim/2)-coords_cor[0][1],(angio_dim/2)-coords_cor[1][1]) , '.r-')    
            #breakpoint()
            #Sag_line_plot[0].set_xdata( np.array([30,50])   )
            #Sag_line_plot[0].set_ydata( np.array([100,120])   )  
            canvas_angio_cor.draw()
            HbO2_output.calc_SS_angle(coords_sag, coords_cor)
            
    cid = fig_angio_cor.canvas.mpl_connect('button_press_event', onclick_cor)
    

    

SSanglemenu = Menu(menubar, tearoff=1)
SSangle_submenu = Menu(SSanglemenu, tearoff=1)
menubar.add_cascade(label="Calc vessel B0 angle", menu=SSanglemenu)
SSanglemenu.add_command(label="Calc angle", command=Calc_SS_angle, accelerator="Control+o")


def popup_help():
    popup = Tk()
    GUI_width = window.winfo_screenwidth()*0.35
    GUI_height = window.winfo_screenheight()*0.25
    popup.geometry( str(int(GUI_width)) +'x'+ str(int(GUI_height)) )
    popup.resizable(True,True)
    popup.wm_title("About me")
    help_str='GUI for calculating oxygen saturation på SBO. \n Useable for PAR/REC philips file.'
    name_str='Mark B. Vestergaard \n mark.bitsch.vestergaard@regionh.dk, \n Functional Imaging Unit \n Department of Clinical Physiology, Nuclear Medicine and PET \n Rigshospitalet, Glostrup, Denmark \n July 2021.' 
    text_title = Label(popup,text=help_str,anchor="w", background='white')
    text_title.pack(side="top", fill="x", pady=10)
    text_name = Label(popup,text=name_str,justify="left",anchor="w")
    text_name.pack(side="top", fill="x", pady=10)
    
    B1 = Button(popup, text="Close", command = popup.destroy)
    B1.pack(side="top", pady=10)
    popup.mainloop()

analysismenu.add_command(label="About", command=popup_help, accelerator="Control+a")
window.bind_all('<Control-Key-a>', func=popup_help)



window.config(menu=menubar) # Insert topmenu
window.mainloop()