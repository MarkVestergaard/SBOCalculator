# SBOCalculator

GUI for analyzing susceptibility-based oximetry MRI data for measuring blood saturation (Jain et al. Journal of Cerebral Blood Flow & Metabolism 2010, Vol.30 (9) 1598–1607, https://doi.org/10.1038/jcbfm.2010.49) 

Written for python 3.8. Tested on SBO data from 3T Philips dSTREAM Achieva MRI and 3T Siemens Biograph mMR hybrid PET/MR system.

Input for Philips scanner data: **.PAR file 

Input for Siemens scanner data: nifti-file converted from dicom using dcmniix (https://github.com/rordenlab/dcm2niix)<n>

Region of interest (ROI) is manual delineated. 
Data saved as csv file.

For in-depth description of analysis see: Vestergaard et al. Cerebral Cortex, Volume 32, Issue 6, 15 March 2022, 1295–1306, doi:https://doi.org/10.1093/cercor/bhab294 or Vestergaard et al. Journal of Cerebral Blood Flow & Metabolism 2019, Vol. 39(5) 834–848, doi:https://doi.org/10.1177/0271678X17737909

Mark B. Vestergaard
Functional Imaging Unit,
Department of Clinical Physiology and Nuclear Medicine
Rigshospitalet Copenhagen, Denmark
mark.bitsch.vestergaard@regionh.dk

<img width="1154" alt="SBOCalculator" src="https://user-images.githubusercontent.com/102877223/161437896-7ece4d42-e316-4440-839f-65236d641e8b.png">
