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

import itertools
import matplotlib.image as pli
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1.inset_locator
import mpl_toolkits.mplot3d
import numpy as np
import os
import pandas
import png
import scipy as sp
import scipy.cluster
import scipy.misc
import scipy.ndimage
import scipy.optimize
import scipy.spatial
import scipy.stats
import skimage
import skimage.draw
import skimage.filters
import skimage.morphology
import skimage.feature
import skimage.measure
import skimage.segmentation
import time

import P4D_help                                 # implementations of various auxiliary functions
reload(P4D_help)

########################################################## plot

def plot(inp): 
    '''Function plots time series of plant and leaf features. Takes feature tables as inputs.'''

    f,F,folder,MainPath,MainCores,AnaTrackStop,AnaTrackRange,AnaTrackMemory,AnaTrackMinlen,AnaAssignMaxDAngles,AnaAssignMaxOverlap,AnaSetupDRER,AnaSetupNightStart,AnaSetupTimeBin,DoTrack,temp=inp     # input parameters
    tempo=time.time()                                                                                                                                                                           # remember starting time
    dirF=folder+'/focus/'                                                                                                       # folder with focus images
    dirN=folder+'/Nums/'                                                                                                        # folder with segmentation data
    dirV=folder+'/Vids/'                                                                                                        # folder for generation of video    
    focus=np.sort(os.listdir(dirF))                                                                                             # list focus images
    nums=np.sort(os.listdir(dirN))                                                                                              # list segmentation data
    cfull=pandas.read_csv(folder+'/data_spreadsheet_full.csv',encoding='utf-8',delimiter=',')                                   # read full feature table
    name_full=cfull.columns                                                                                                     
    full=cfull.as_matrix()                                                                                                      
    J,QA=full.shape
    I=AnaTrackStop#J-10#
    II=np.arange(I)  
    cfold=pandas.read_csv(folder+'/data_spreadsheet_fold.csv',encoding='utf-8',delimiter=',')                                   # read foled feature table
    name_fold=cfold.columns
    fold=cfold.as_matrix()   
    H,QT=fold.shape    
    chist=pandas.read_csv(folder+'/data_spreadsheet_hist.csv',encoding='utf-8',delimiter=',')                                   # read histogram feature table
    name_hist=chist.columns
    hist=chist.as_matrix()   
    quants,names,units,[HX,HH,Q0,Q1,lx,ly,dxy]=np.load(folder+'/data_analyze_quants.npy')                                       # load parameters from track analysis
    [S,B,A],beta,alpha,alphi,xx=np.load(folder+'/data_analyze_assignment.npy')                                                  # load parameters from track assignment
    [ummm,ucmx,upph,udas,uhou,ummx,udeg,uper,uxxx],[qare,qcirc,qcomp,qcurv,qasra,qxoc,qxbt,qzoc,qzbt,qzlr,qpox,qpoy,qpoz]=np.load(folder+'/data_analyze_units.npy')                             # load unit labels and feature numbers 
    leafs=['I','II']+list([str(b) for b in range(1,B-1)])                                                                       # label cotyledons and first six leaves
    cobs=[plt.cm.jet(1.0*b/(B-1.0)) for b in range(B)]                                                                          # color cotyledons and first six leaves
    days=full[:I,0]                                                                                                              # get time of frames in days
    darks=full[:I,2]                                                                                                             # get dark information            
    midnights=full[:I,3]                                                                                                         # get midnight information
    unit_full=np.hstack(units[:Q0]+[[units[q] for q in range(Q0,Q0+Q1)] for b in range(B)])                                     # set units for full time series
    unit_fold=np.hstack([[u for app in ['_avg','_std']] for u in unit_full])                                                    # set units for folded time series
    wh=[ui for ui,u in enumerate(name_fold) if '_2Darea' in u or '_3Darea' in u]                                                # indicate area-based features
    unit_fold[wh]=upph                                                                                                          # change the unit for these features
    unit_hist=unit_fold                                                                                                         # set units for histogram data
            
########################################################## 3D plot 
       
    fig=plt.subplot(111,projection='3d')                                                                                        # plot leaf tracks in 3D
    fig.view_init(elev=40,azim=-40)    
    for b in range(B):         
        x=P4D_help.sliding_median(full[:I,qpox+b*Q1],window=20)
        y=P4D_help.sliding_median(full[:I,qpoy+b*Q1],window=20)
        plt.plot(y[:I],x[:I],II[:I],color=cobs[b],lw=2)    
    ax=plt.axis()    
    plt.xlabel('$x$'+ucmx)
    plt.ylabel('$y$'+ucmx)  
    x0,x1=int(ax[0]*dxy/10.0),int(ax[1]*dxy/10.0)+2
    y0,y1=int(ax[2]*dxy/10.0),int(ax[3]*dxy/10.0)+2
    plt.axis(np.array([x0,x1-0.999,y0,y1-0.999])*10.0/dxy)
    plt.xticks(np.arange(x0,x1)*10.0/dxy,range(x0,x1))
    plt.yticks(np.arange(y0,y1)*10.0/dxy,range(y0,y1))         
    wh=np.where(np.diff(days))[0][:-2]
    wh=wh[days[wh[0]+1]%2::2]+1  
    fig.set_zticks(wh)
    fig.set_zticklabels(days[wh].astype('int'))
    fig.set_zlabel('time'+udas)
    plt.tight_layout()
    plt.savefig(folder+'/fig_analyze_tracks.svg')
    #plt.show()    
       
############################################################### full

    for q in range(Q0):                                                                                                         # plot all plant feature over the full time
        plt.clf()
        sm=P4D_help.sliding_median(full[:I,q],window=2)
        plt.plot(II,sm,color=cobs[0],lw=2)
        P4D_help.xlabel_full(II,darks,midnights,days,xlabel='time'+udas,alpha=0.2)    
        plt.ylabel(name_full[q]+unit_full[q])
        plt.tight_layout()
        plt.savefig(folder+'/fig_plant_full_'+name_full[q]+'.svg')
        #plt.show()  
         
    for q in range(Q1):                                                                                                         # plot all leaf feature over the full time
        plt.clf()
        for b in range(B):            
            sm=P4D_help.sliding_median(full[:I,Q0+b*Q1+q],window=2)
            plt.plot(II,sm,color=cobs[b],lw=2)
        P4D_help.xlabel_full(II,darks,midnights,days,xlabel='time'+udas,alpha=0.2)    
        plt.ylabel(name_full[Q0+q]+unit_full[Q0+q][3:])
        plt.tight_layout()
        plt.savefig(folder+'/fig_leaf_full_'+name_full[Q0+q][3:]+'.svg')
#        plt.show()  
               
############################################################### fold    

    for q in range(Q0):                                                                                                         # plot all plant feature as 24-hour averages
        plt.clf()
        avg=P4D_help.sliding_median(fold[:,2*q+0],window=2)
        std=P4D_help.sliding_median(fold[:,2*q+1],window=2)
        if(np.nansum(std)>0):
            plt.fill_between(HX,avg-std,avg+std,color=cobs[0],lw=0,alpha=0.2)
        plt.plot(HX,avg,color=cobs[0],lw=2)
        ax=P4D_help.xlabel_fold(HH,AnaSetupTimeBin,xlabel='time'+uhou,alpha=0.2)   
        plt.ylabel(name_fold[2*q][:-4]+unit_fold[2*q])
        plt.tight_layout()
        plt.savefig(folder+'/fig_plant_fold_'+name_fold[2*q][:-4]+'.svg')
#        plt.show()       
                
    for q in range(Q1):                                                                                                       # plot all leaf feature as 24-hour averages
        plt.clf()
        for b in range(B):
            avg=P4D_help.sliding_median(fold[:,2*Q0+2*b*Q1+2*q+0],window=2)
            std=P4D_help.sliding_median(fold[:,2*Q0+2*b*Q1+2*q+1],window=2)
            if(np.nansum(std)>0):
                wh=np.logical_or(avg!=avg,std!=std)
                avg[wh]=0.0
                std[wh]=0.0
                plt.fill_between(HX,avg-std,avg+std,color=cobs[b],lw=0,alpha=0.2)
            plt.plot(HX,avg,color=cobs[b],lw=2)
        ax=P4D_help.xlabel_fold(HH,AnaSetupTimeBin,xlabel='time'+uhou,alpha=0.2)       
        plt.ylabel(name_fold[2*Q0+2*q][3:-4]+unit_fold[2*Q0+2*q])
        plt.tight_layout()
        plt.savefig(folder+'/fig_leaf_fold_'+name_fold[2*Q0+2*q][3:-4]+'.svg')
#        plt.show()  
        
############################################################### hist 
        
    for q in range(Q0):                                                                                                         # plot all plant feature as histograms for changes at day and night
        plt.clf() 
        props=[[] for i in range(2)]        
        for d in range(2):
            prop=hist[:,2*q+d]
            props[d]=prop[prop==prop]
            avg=sp.stats.nanmean(props[d])
            std=sp.stats.nanstd(props[d],bias=1)
            plt.errorbar(1+d,avg,yerr=std,ls='.',ecolor=cobs[0],elinewidth=4,capsize=10,capthick=4,alpha=1.0,zorder=1)        
            plt.bar(1+d,avg+99.0,width=0.9,lw=4,align='center',ec=cobs[0],color='white',alpha=1.0,bottom=-99.0,zorder=2)
        tt=sp.stats.ttest_ind(props[0],props[1])[1]
        ax=plt.axis()
        plt.text(1.5,ax[3]*0.9,'p='+'%.1e'%tt,ha='center')        
        plt.xticks([1,2],['day','night'])    
        plt.ylabel(name_hist[2*q][:-4]+unit_hist[2*q])
        plt.tight_layout()
        plt.savefig(folder+'/fig_plant_hist_'+name_hist[2*q][:-4]+'.svg')    
#        plt.show() 
 
    for q in range(Q1):                                                                                                         # plot all leaf feature as histograms for changes at day and night
        plt.clf()        
        tts=[]
        for b in range(B):
            props=[[] for i in range(2)]  
            for d in range(2):
                qx=2*Q0+2*b*Q1+2*q
                prop=hist[:,qx+d]
                props[d]=prop[prop==prop]
                avg=sp.stats.nanmean(props[d])
                std=sp.stats.nanstd(props[d],bias=1)
                plt.errorbar(2*b+d+1,avg,yerr=std,ls='.',ecolor=cobs[b],elinewidth=4,capsize=10,capthick=4,alpha=1.0,zorder=1)        
                plt.bar(2*b+d+1,avg+99.0,width=0.9,lw=4,align='center',ec=cobs[b],color='white',alpha=1.0,bottom=-99.0,zorder=2) 
            if(len(props[0])>0 and len(props[1])>0):
                ttp=sp.stats.ttest_ind(props[0],props[1])[1]
            else:
                ttp=1.0
            tts.append(ttp)
        ax=plt.axis()
        for b in range(B):
            plt.text(1.5+b,ax[3]*0.9,'p='+'%.1e'%tts[b],ha='center')      
        plt.xticks(range(1,1+2*B),['day','night']*B)        
        plt.ylabel(name_fold[2*Q0+2*q][3:-4]+unit_fold[2*Q0+2*q])
        plt.tight_layout()
        plt.savefig(folder+'/fig_leaf_hist_'+name_fold[2*Q0+2*q][3:-4]+'.svg')
#        plt.show()  

###################################################### make movies    

    print 'plot:','folder',f+1,F,'movie',0,I                                                                                    # adjust and plot leaf segmentation with colors representing leaf numbers
    temp=time.time()                                                                                                            # remember starting time
    ix=list(alphi)                                                                                                              # get list of assigned tracks
    table=np.load(folder+'/data_analyze_table.npy')                                                                             # load tracking table
    for ii,i in enumerate(II[:0]):                                                                                              # for all images
        print 'plot:','folder',f+1,F,'movie',ii+1,I                                                                             # print progress                                        
        im=1.0*np.vstack(itertools.imap(np.uint8,png.Reader(folder+'/Mats/mats_t='+str(i).zfill(6)+'.png').read()[2]))[::-1,::4]# load segmented images
        imc=im.copy()*np.nan                                                                                                    # create auxilliary duplicate
        L=int(im.max())                                                                                                         # get number of leafs
        tt=table[table[:,1]==i]                                                                                                 # get all tracks at current time    
        for l in range(0,L):                                                                                                    # for each leaf do       
            iml=np.nan                                                                                                          # set dummy label                                                                                                        
            track=tt[tt[:,4]==l][0,0]                                                                                           # get track for current leaf               
            if(track in ix):                                                                                                    # if track is among the assigned ones     
                il=np.where(xx[ix.index(track)])[0]                                                                             # get the leaves the track has been assigned to
                if(len(il)>0):                                                                                                  # if it has been assigned, change the leaf label
                    iml=il[0]
            imc[im==track+1]=iml                                                                                                # set the leaf label
        pli.imsave(dirV+'vids_t='+str(ii+1).zfill(6)+".jpg",1.0*imc/(B-1.0),cmap='jet')                                         # save the relabeled segmentation image
    #subprocess.Popen(['ffmpeg','-y','-r','20','-b','1000','-i',dirV+'vids_t=%06d.jpg',folder+'/fig_video.mp4'],stdin=subprocess.PIPE,stderr=subprocess.PIPE)   # create a movie from the relabeled segmentation images  
    print 'plot:','folder',f+1,F,'movie',(time.time()-temp)/60.0,'min'  
       
    S=F                                                                                                                             # compute number of folders to process
    s=f                                                                                                                             # compute number of current image
    dt=time.time()-tempo                                                                                                            # compute duration of current analysis step
    print 'plot:','folder',f+1,F,'time spent',dt,'seconds','time remaining',(S-s)*dt/(3600.0*MainCores),'hours'                     # print computation time statistics
    
    return 0
        

        
    

