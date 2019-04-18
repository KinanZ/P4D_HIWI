###################################################### license

#    Copyright 2014 David Breuer and Federico Apelt 
#    breuer@mpimp-golm.mpg.de and apelt@mpimp-golm.mpg.de
#    http://mathbiol.mpimp-golm.mpg.de/phytotyping4D/
#
#    This file is part of Phytotyping4D.
#    
#    Phytotyping4D is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    
#    Phytotyping4D is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    
#    You should have received a copy of the GNU General Public License
#    along with Phytotyping4D. If not, see <http://www.gnu.org/licenses/>.

###################################################### imports

import multiprocessing
import numpy as np
import os
import sys
import time

import P4D_help                                 # implementations of various auxiliary functions
reload(P4D_help)
import P4D_segment                              # function for segmentation of leafs and plants and computation of positions and shape factors
reload(P4D_segment)
import P4D_analyze                              # function for tracking of leaves, leaf number assignment and export of leaf-specific time series data
reload(P4D_analyze)
import P4D_track                                # implementation of Crocker-Grier tracking algorithm by Thomas A. Caswell under GLP3
reload(P4D_track)
import P4D_plot                                 # function for automated plotting of plant and leaf features for one plant
reload(P4D_plot)
import P4D_multi                                # function for generation of customized summary plots including multiple plants and genotypes
reload(P4D_multi)

if __name__=='__main__':                        # freeze support for multiprocessing under Windows
    multiprocessing.freeze_support()

###################################################### folders and parameters

MainPath='/home/kinan/Desktop/HIWI/test_Machine'           # path with image time series
MainFolder='140601_1900_Col0_vs_pgm'            # folder of experiment to analyze, second number is start of night, e.g., 19:00
MainCores=4                                     # number of cores
MainNth=1                                       # analyze only every nth subfolder
MainOverwrite=0                                 # Boolean parameter to control wether already generated output is overwritten or not

DoSegment=1                                     # do segmentation (plants and individual leaves)
DoAnalyze=1                                     # do analysis (track leaves, assign numbers, and compute features)
DoPlot=1                                        # do single plots (different automated plots of all plant and leaf features)
DoMulti=1                                       # do multi comparison (combine data of multiple folders for different genotypes and generate customized plots for publication)

SegBGupper=25                                   # upper threshold for image background
SegFGlower=80                                   # lower threshold for image foreground
SegSigmaGauss=50.0                              # sigma of Gaussian filter for smoothing depth image
SegSigmaCanny=2.0                               # sigma of Canny filter for detection of leaf edges             
SegThresSlope=0.1                               # slope of radially increasing threshold to find watershed seeds from distance transformed image
SegThresAbsci=20.0                              # abscissa of radially increasing threshold to find watershed seeds from distance transformed image
SegRadiusOriginHeight=200.0                     # size of disk for determining height of origin in pixels
SegRadiusLeafHeight=20.0                        # size of disk for determining height of leaf positions in pixels
SegRadiusStemEraser=20.0                        # size of disk for removing leaf petioles

AnaTrackOver=0                                  # overwrite tracking results
AnaTrackStop=900                                # stop analyzing at given frame
AnaTrackRange=30.0                              # maximal linkage distance for connecting leaf centers in pixels, default=30
AnaTrackMemory=20                               # maximal frame gap to be bridged in number of frames, default=20
AnaTrackMinlen=10                               # minimal length of tracks to be considered, default=10
AnaAssignMaxDAngles=30.0                        # maximal angle difference between leaf position and track for assignment
AnaAssignMaxOverlap=10                          # maximal number of overlapping frame for assignment of different tracks to same leaf position
AnaSetupDRER=1.0                                # time difference for computation of RER in hours    
AnaSetupNightStart=20.03                        # time of nightfall in hours # 19.03 for pex/elf # 20.03 for the rest
AnaSetupTimeBin=10.0                            # time bin for reasonable 24h binning in minutes

############################################################### general setup

exceptions=['Raw','Scan','XXX','YYY','.tif','.tiff','.png','.jpg','.svg','.txt','.npy','.db','.xlsx','.sh']   # ignore folder containing these strings
project=MainPath+'/'+MainFolder                                                         # get project folder for output
folders=P4D_help.generate_subfolders(project,exceptions)                                # generate list of folders for all conditions and experiments
folders=folders[::MainNth]                                                              # use every Nth folder only
F=len(folders)                                                                          # get number of folders to process
geno=np.array([p.split('/')[-2] for p in folders])                                      # get genotype for each folder
G=len(set(geno))                                                                        # get number of genotypes

AnaSetupNightStart=0.01*int(MainFolder.split('_')[1])                                   # time of nightfall in hours
cores=multiprocessing.cpu_count()                                                       # get number of available cores
MainCores=min(MainCores,cores)                                                          # limit number of cores to use 
if(cores<3):                                                                            # if only 2 cores are available on the local PC, use only one to smooth working
    MainCores=1
pool=multiprocessing.Pool(processes=MainCores)                                          # set number of cores used for parallilization
sys.stdout.flush()                                                                      # enable printing during parallel processing
print 'Phytotyping4D:','analyzing',F,'plant movies using',MainCores,'/',cores,'cores'   # print number of folders to process and cores used
#sys.exit()

###################################################### segment

jobs=[]                                                                     # list for queued jobs
temp=time.time()                                                            # store starting time
for f,folder in enumerate(folders):                                         # for each folder do
    dirD=folder+'/depth/'                                                   # folders with depth images and focus images
    dirF=folder+'/focus/'  
    dirM=folder+'/Mats/'                                                    # folders for segmenation output, extracted features etc.
    dirN=folder+'/Nums/'
    dirP=folder+'/Pics/' 
    dirS=folder+'/Seed/' 
    dirV=folder+'/Vids/'      
    for i in [dirM,dirN,dirP,dirS,dirV]:                                    # generation output folders if not present
        if(not os.path.isdir(i)): os.mkdir(i)   
    depth=np.sort(os.listdir(dirD))                                         # get depth images
    focus=np.sort(os.listdir(dirF))                                         # get focus images
    I=len(depth)                                                            # compute number of images in current folder
    for i in range(I):                                                      # for each image do
        name=dirN+'nums_t='+str(i).zfill(6)+'.npy'                          # name of segmentation output
        if((not os.path.isfile(name) or MainOverwrite) and DoSegment):      # add to queue if no output present or overwrite is true
            inp=f,F,i,I,depth[i],focus[i],dirD,dirF,dirM,dirN,dirP,dirS,dirV,MainPath,MainCores,SegBGupper,SegFGlower,SegSigmaGauss,SegSigmaCanny,SegThresSlope,SegThresAbsci,SegRadiusOriginHeight,SegRadiusLeafHeight,SegRadiusStemEraser,AnaTrackOver,temp
            #sys.exit()            
            jobs.append((P4D_segment.segment,(inp)))                        # add segmentation to queue 
            
print 'segment:','queued',len(jobs)                                         # print number of images to segment
pool.map_async(P4D_help.doexception,tuple(jobs),chunksize=1).get(9999999)   # start parallel processing
print 'segment:','done'                                                     # print done
#sys.exit() 

###################################################### analyze

jobs=[]                                                                     # list for queued jobs
temp=time.time()                                                            # store starting time
for f,folder in enumerate(folders):                                         # for each folder do
    name=folder+'/data_spreadsheet_hist.csvX'                                # name of analysis output
    if((not os.path.isfile(name) or MainOverwrite) and DoAnalyze):          # add to queue if no output present or overwrite is true
        inp=f,F,folder,MainPath,MainCores,AnaTrackStop,AnaTrackRange,AnaTrackMemory,AnaTrackMinlen,AnaAssignMaxDAngles,AnaAssignMaxOverlap,AnaSetupDRER,AnaSetupNightStart,AnaSetupTimeBin,AnaTrackOver,temp
        #sys.exit()
        jobs.append((P4D_analyze.analyze,(inp)))                            # add analysis to queue 

print 'analyze:','queued',len(jobs)                                         # print number of movies to analyze
pool.map_async(P4D_help.doexception,tuple(jobs),chunksize=1).get(9999999)   # start parallel processing
print 'analyze:','done'                                                     # print done
#sys.exit()   

###################################################### plot

jobs=[]                                                                     # list for queued jobs
temp=time.time()                                                            # store starting time
for f,folder in enumerate(folders):                                         # for each folder do
    name=folder+'/fig_analyze_tracks.svgX'                                   # name of plot output
    if((not os.path.isfile(name) or MainOverwrite) and DoPlot):             # add to queue if no output present or overwrite is true
        inp=f,F,folder,MainPath,MainCores,AnaTrackStop,AnaTrackRange,AnaTrackMemory,AnaTrackMinlen,AnaAssignMaxDAngles,AnaAssignMaxOverlap,AnaSetupDRER,AnaSetupNightStart,AnaSetupTimeBin,AnaTrackOver,temp
        #sys.exit()
        jobs.append((P4D_plot.plot,(inp)))                                  # add plot to queue 

print 'plot:','queued',len(jobs)                                            # print number of movies to plot
pool.map_async(P4D_help.doexception,tuple(jobs),chunksize=1).get(9999999)   # start parallel processing
print 'plot:','done'                                                        # print done
#sys.exit()   

###################################################### multi

jobs=[]                                                                     # list for queued jobs
temp=time.time()                                                            # store starting time
for i in range(1):                                                          # for the chosen project do
    name=project+'/fig5a_leafareas.svgX'                                     # name of multi output
    if((not os.path.isfile(name) or MainOverwrite) and DoMulti):            # add to queue if no output present or overwrite is true    
        inp=project,folders,geno,G,MainPath,MainCores,AnaTrackStop,AnaTrackRange,AnaTrackMemory,AnaTrackMinlen,AnaAssignMaxDAngles,AnaAssignMaxOverlap,AnaSetupDRER,AnaSetupNightStart,AnaSetupTimeBin,AnaTrackOver,temp
        #sys.exit()
        jobs.append((P4D_multi.multi,(inp)))                                # add multi to queue 
    
print 'multi:','queued',len(jobs)                                           # print number of projects to multi plot
pool.map_async(P4D_help.doexception,tuple(jobs),chunksize=1).get(9999999)   # start parallel processing
print 'multi:','done'                                                       # print done
