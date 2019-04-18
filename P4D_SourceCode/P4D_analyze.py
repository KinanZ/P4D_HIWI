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

import copy
import csv
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import png
import scipy as sp
import scipy.cluster
import scipy.misc 
import scipy.ndimage
import scipy.optimize
import time

import P4D_help                                 # implementations of various auxiliary functions
reload(P4D_help)
import P4D_track                                # implementation of Crocker-Grier tracking algorithm by Thomas A. Caswell under GLP3
reload(P4D_track)

########################################################## analyze

def analyze(inp): 
    '''Function tracks individual leaves and automatically assigns leaf numbers. Takes leaf positions and features as inputs.''' 
    
    f,F,folder,MainPath,MainCores,AnaTrackStop,AnaTrackRange,AnaTrackMemory,AnaTrackMinlen,AnaAssignMaxDAngles,AnaAssignMaxOverlap,AnaSetupDRER,AnaSetupNightStart,AnaSetupTimeBin,AnaTrackOver,temp=inp     # input parameters
    tempo=time.time()                                                                                                                                                                           # remember starting time
    dirF=folder+'/focus/'
    dirN=folder+'/Nums/'
    files=np.sort(os.listdir(dirF))                                                                                                                                                             # list focus images
    images=np.sort(os.listdir(dirN))                                                                                                                                                            # list extracted data
    I=len(images)                                                                                                                                                                               # analyse up to this image
    II=np.arange(I)       
    data=[]    
    for image in images:                                                                                                                                                                        # load extracted data
        data.append(np.load(dirN+image))  
    lx,ly,dxy,L,total_size,total_area,total_touch,total_perim,total_chull=data[0][0]                                                                                                            # get basic image parameters
    LL=[data[i][0][3] for i in range(I)]                                                                                                                                                        # get number of leaves per image
    NL=np.sum(LL)                                                                                                                                                                               # compute total number of leaves in movie       
    dxy=np.median([data[i][0][2] for i in range(I)])                                                                                                                                            # compute median pixelsize over the duration of the movie
    origin=np.median([data[i][2] for i in range(I)],0)                                                                                                                                          # compute median origin over the duration of the movie
    B=8                                                                                                                                                                                         # set maximal number of leafs to analyze
    leafs=['I','II']+list([str(b) for b in range(1,B-1)])                                                                                                                                       # set labels for cotyledons and leaves
    cobs=[plt.cm.jet(1.0*b/(B-1.0)) for b in range(B)]                                                                                                                                          # set colors for cotyledons and leaves
    
########################################################## leaf tracking and labeling GC
    
    nama=folder+'/data_analyze_table.npy'
    namc=folder+'/data_analyze_tracks.npy'    
    temp=time.time()
    if(not os.path.isfile(namc) or AnaTrackOver):                                                                              # if leaves have not been tracked yet, or overwrite is true
        levels=[[P4D_track.PointND(i,data[i][4][l][4][0:2][::-1]) for l in range(LL[i])] for i in range(I)]                     # create list with leaf center positions for tracking
        hg=lambda: P4D_track.Hash_table((ly,lx),1.0)                                                                            # create hash table for Croker-Grier tracking with extent of tracking region and time step
        tracks=P4D_track.link(levels,hash_generator=hg,search_range=AnaTrackRange,memory=AnaTrackMemory)                        # track leaf centers
        np.save(namc,tracks)                                                                                                    # save tracking data
        tracks=np.load(namc)           
        table=np.zeros((NL,5))                                                                                                  # create table with reordered tracking data, containing track label, time, x, y, leaf number   
        t0=0
        for ti,track in enumerate(tracks):                                                                                      # for each track do
            t1=len(track)                                                                                                       # get length if track
            tt=range(t1)
            table[t0:t0+t1,0]=ti                                                                                                # set track label for associated leaves
            table[t0:t0+t1,1]=[track.points[t].t for t in tt]                                                                   # set times for associated leaves
            table[t0:t0+t1,2]=[track.points[t].pos[0] for t in tt]                                                              # set x-positions for associated leaves
            table[t0:t0+t1,3]=[track.points[t].pos[1] for t in tt]                                                              # set y-positions for associated leaves
            table[t0:t0+t1,4]=[np.argmax(np.sum([p.pos==point.pos for p in levels[point.t]],axis=1)) for point in track.points] # set segmentation number for associated leaves
            t0=t0+t1
        np.save(nama,table)                                                                                                     # save tracking table
    else:                                                                                                                       # load tracking data and table otherwise
        tracks=np.load(namc)     
        table=np.load(nama)          
    T=int(table[:,0].max())+1                                                                                                   # set number of tracks
    TT=range(T)                                                                             
    print 'analyze:','folder',f+1,F,'tracking',(time.time()-temp)/60.0,'min'                                                    # print duration of tracking
 
###################################################### time series analysis
      
    ummm,ucmx,upph,udas,uhou,ummx,udeg,uper,uxxx=" [$\mathrm{mm}^2$]"," [$\mathrm{cm}$]"," [$\%/\mathrm{h}$]"," [$\mathrm{DAS}$]"," [$\mathrm{h}$]"," [$\mathrm{mm}$]"," [$^{\circ}$]"," [$\%$]",""     # set unit labels 
    
    qare,qcirc,qcomp,qcurv,qasra,qxoc,qxbt,qzoc,qzbt,qzlr,qpox,qpoy,qpoz=24,26,27,28,31,33,36,38,41,42,43,44,45                                                                                         # set property indices
    np.save(folder+'/data_analyze_units.npy',[[ummm,ucmx,upph,udas,uhou,ummx,udeg,uper,uxxx],[qare,qcirc,qcomp,qcurv,qasra,qxoc,qxbt,qzoc,qzbt,qzlr,qpox,qpoy,qpoz]])                                   # save unit labels and property indices
        
    # data                                                                                      # structure of segmentation data
    #     0    1     2    3    4       
    # [info,origin,com,quants,coords,[0]]
    #
    # info
    #  0  1  2   3      4         5        6             7               8
    # [lx,ly,dxy,L,total_size,total_area,total_touch,total_perim,total_chull]       
    #   
    # quants
    #   0      1    2      3    4
    # [touch,perim,chull,size,area]  
    #
    # coords
    #    0      1       2     3      4
    # ['base','tip','right','left','center']     
    #
    # quantities
    #        0      1     2         3         4          5           6            7           8            9            10                11               12                    13                  14        15           16           17     18     19        20                    21               22                       23       24           25                26              27           28                 29          30              31            32                      33              34            35           36                       37            38               39         40             41             42           43       44      45
    names=['day','hour','dark','midnight','AnaSetupTimeBin','contime','pixelsize','plant_leafs','plant_2Darea','plant_3Darea','plant_touch','plant_circularity','plant_compactness','plant_asymmetry','origin_x','origin_y','origin_z','com_x','com_y','com_z','reliable_fraction','reliable_average','reliable_weighted','leaf_2Darea','leaf_3Darea','leaf_touch','leaf_circularity','leaf_compactness','leaf_curvature','leaf_length','leaf_width','leaf_aspectratio','leaf_distance','xy_origin_center','xy_origin_base','xy_origin_tip','xy_base_tip','xy_left_right','z_origin_center','z_origin_base','z_origin_tip','z_base_tip','z_left_right','posi_x','posi_y','posi_z']          # compute plant and leaf features                 
    units=[udas,  uhou, uxxx,   uxxx,      uxxx,       uxxx,     ummx,        uxxx,         ummm,       ummm,        uxxx,           uxxx,                uxxx,             ummx,             ummx,        ummx,   ummx,        ummx,  ummx,  ummx,     uper,               udeg,               udeg,               ummm,     ummm,         uxxx,       uxxx,           uxxx,                   ummx,       ummx,           ummx,       uxxx,               ummm,           udeg,               udeg,           udeg,          udeg,          udeg,          udeg,             udeg,            udeg,          udeg,          udeg,      ummx,     ummx,    ummx        ]                   # set corresponding units
    H=int(24.0*60.0/AnaSetupTimeBin)                                                                                            # compute number of time bins for 24-hour averages
    ns=(60*np.mod(int(AnaSetupNightStart+12.00),24)+int(np.mod(AnaSetupNightStart+12.00,1.0)*100))/int(AnaSetupTimeBin)         # compute bin of start of night
    HH=range(H)[ns:]+range(H)[:ns]                                                                                              # rearrange bins to start with start of day
    HX=range(H)
    Q=len(names)                                                                                                                # total number of properties
    Q0=23                                                                                                                       # number of plant properties
    Q1=Q-Q0   
    QQ=range(Q)    
    quants=[[[] for i in II] for q in QQ]                                                                                       # create container for all properties    
    for i in range(I):                                                                                                          # for each image compute plant properties        
        day=int(files[i][-18:-16])                                                                                              # compute plant time info
        quants[0][i].append(day)                                                       
        hour=float(files[i][-15:-10])       
        quants[1][i].append(hour)
        if(hour>AnaSetupNightStart or hour<np.mod(AnaSetupNightStart+12.00,24.00)):
            dark=1
        else:
            dark=0
        quants[2][i].append(dark)    
        if(quants[1][i][-1]<1.00 and quants[1][i-1][-1]>23.00):
            midnight=1
        else:
            midnight=0
        quants[3][i].append(midnight)   
        time_bin=(60*np.mod(int(hour),24)+int(np.mod(hour,1.0)*100))/int(AnaSetupTimeBin)   
        quants[4][i].append(time_bin)         
        quants[5][i].append((24*60)*day+60*int(hour)+100*np.mod(hour,1.0))        
        ori=data[i][1]                                                                                                          # compute plant features 
        com=data[i][2]
        quants[6][i].append(data[i][0][2])
        quants[7][i].append(data[i][0][3])  
        quants[8][i].append(data[i][0][4]*dxy*dxy)                                          
        quants[9][i].append(data[i][0][5])        
        quants[10][i].append(data[i][0][6])                                               
        quants[11][i].append(4.0*np.pi*data[i][0][4]/data[i][0][7]**2)
        quants[12][i].append(1.0000000*data[i][0][4]/data[i][0][8])  
        quants[13][i].append(P4D_help.distance(ori,com,dxy))
        quants[14][i].append(data[i][1][0])                                                                                     # compute plant origin                                            
        quants[15][i].append(data[i][1][1])
        quants[16][i].append(data[i][1][2])    
        quants[17][i].append(data[i][2][0])                                                                                     # compute plant middle                       
        quants[18][i].append(data[i][2][1])
        quants[19][i].append(data[i][2][2])
        for l in range(LL[i]):                                                                                                  # for each image compute leaf properties
            origin=data[i][2]                                                                                                   # compute leaf positions                                               
            base=data[i][4][l][0]
            tip=data[i][4][l][1]
            right=data[i][4][l][2]
            left=data[i][4][l][3]
            center=data[i][4][l][4]                           
            dist,nearest=P4D_help.pnt2line(center,left,right)
            rcenter,rbase,rtip,rleft,rright=P4D_help.vec_scale(np.vstack([center,base,tip,left,right]),dxy)   
            v0,v1,v2=rcenter-rbase,rright-rleft,rtip-rbase              
            if(np.sum(np.abs(v0))>0.0 and np.sum(np.abs(v1))>0.0 and np.sum(np.abs(v2))>0.0):                                   # compute distance of leaf center from left-right axis
                vres=P4D_help.pnt2plane(v0,v1,v2)
            else:
                vres=[-1.0,-1.0,-1.0]#
            length=P4D_help.distance(base,tip,dxy)
            width=P4D_help.distance(left,right,dxy)            
            quants[23][i].append(data[i][3][l][3]*dxy*dxy)                                                                      # compute leaf features                                    
            quants[24][i].append(data[i][3][l][4])
            quants[25][i].append(data[i][3][l][0])           
            quants[26][i].append(4.0*np.pi*data[i][3][l][3]/data[i][3][l][1]**2)                              
            quants[27][i].append(1.0000000*data[i][3][l][3]/data[i][3][l][2])              
            quants[28][i].append(vres[2])
            quants[29][i].append(length)  
            quants[30][i].append(width)  
            quants[31][i].append(1.0*width/length)  
            quants[32][i].append(P4D_help.distance(origin,center,dxy))            
            quants[33][i].append(P4D_help.angle_xy((origin-center)[0:2]))                                                       # compute leaf xy-angles
            quants[34][i].append(P4D_help.angle_xy((origin-base)[0:2]))
            quants[35][i].append(P4D_help.angle_xy((origin-tip)[0:2]))   
            quants[36][i].append(P4D_help.angle_xy((base-tip)[0:2]))
            quants[37][i].append(P4D_help.angle_xy((right-left)[0:2]))            
            quants[38][i].append(-P4D_help.angle_z(origin,center,dxy))                                                          # compute leaf z-angle
            quants[39][i].append(-P4D_help.angle_z(origin,base,dxy))
            quants[40][i].append(-P4D_help.angle_z(origin,tip,dxy))  
            quants[41][i].append(-P4D_help.angle_z(base,tip,dxy))              
            quants[42][i].append(-P4D_help.angle_z(right,left,dxy))                                                  
            quants[43][i].append(center[0])                                                                                     # compute leaf center   
            quants[44][i].append(center[1])       
            quants[45][i].append(center[2])         
        quants[20][i].append(np.sum([(1.0*quants[41][i][l]>20.0) for l in range(LL[i])])/np.sum([1.0 for l in range(LL[i])]))                                   # reliability: compute fraction of leaves with base-tip z-angle 
        quants[21][i].append(np.sum([(1.0*quants[41][i][l]) for l in range(LL[i])])/np.sum([1.0 for l in range(LL[i])]))                                        # compute average base-tip z-angle 
        quants[22][i].append(np.sum([(1.0*quants[41][i][l]*quants[23][i][l]) for l in range(LL[i])])/np.sum([1.0*quants[23][i][l] for l in range(LL[i])]))      # compute area-weighted average base-tip z-angle 
    np.save(folder+'/data_analyze_quants.npy',[quants,names,units,[HX,HH,Q0,Q1,lx,ly,dxy]])                                                                     # save plant and leaf features

    
###################################################### reliability  
    
    days=np.array([quants[0][i][0] for i in II])                                                            # set day for each image
    hours=np.array([quants[1][i][0] for i in II])                                                           # set hour 
    darks=np.array([quants[2][i][0] for i in II])                                                           # set if dark 
    midnights=np.array([quants[3][i][0] for i in II])                                                       # set if midnight
    bins=np.array([quants[4][i][0] for i in II])                                                            # set time bin
    contime=np.array([quants[5][i][0] for i in II])                                                         # set minute
    
    angi=np.zeros((100,I))                                                                                  # auxilliary array for plotting of reliability information
    ams=np.zeros(I)
    for i in range(I):                                                                                      # for each image do
        an=np.array([(quants[qzbt][i][l]) for l in range(LL[i])])                                           # get base-tip z-angles for each leaf
        ai=np.argsort(an)                                                                                   # get indices of sorted angles 
        aa=np.cumsum([0.0]+list(np.array([quants[qare][i][l] for l in range(LL[i])])[ai]))                  # compute cummulative sums of areas ordered parallel to angles
        for l in range(LL[i]):                                                                              # for each leaf do
            amin=int(100.0*aa[l+0]/aa[-1]+0.5)                                                              # compute upper fraction of cummulative leaf area
            amax=int(100.0*aa[l+1]/aa[-1]+0.5)                                                              # compute lower fraction of cummulative leaf area
            angi[amin:amax,i]=(an[ai[l]])                                                                   # set leaf fraction to corresponding leaf angle
        ams[i]=quants[22][i][0]                                                                             # compute area-weighted average base-tip z-angle
    
    plt.clf()                                                                                               # plot reliability angles
    plt.imshow(angi,origin='lower',interpolation='nearest',vmin=-25.0,vmax=25.0,aspect='auto')  
    cb=plt.colorbar(orientation='vertical',aspect=40,format="%.f",ticks=range(-25,26,5))
    cb.set_label('angle [$^{\circ}$]')
    cb.ax.yaxis.set_ticks_position('left')    
    for cbl in cb.ax.get_yticklabels():
        cbl.set_horizontalalignment('left')  
        cbl.set_x(-1.8)     
    plt.plot(II,100.0/50.0*P4D_help.sliding_median(ams,window=3)+50.0,lw=2,color='white',alpha=1.0) 
    plt.plot([0,I],[50,50],lw=2,color='white',alpha=1.0,ls='--')
    P4D_help.xlabel_full(II,darks,midnights,days,xlabel='time'+udas,alpha=0.2)
    plt.xlim([0,I])
    plt.ylim([0,100.00000001])
    plt.ylabel('plant area [$\%$]')
    plt.savefig(folder+'/fig_plant_full_reliability.svg')
    np.save(folder+'/data_analyze_full_reliability.npy',[angi,ams])   
#    plt.show()
    
    ams_avg,ams_std,ams_davg,ams_navg=P4D_help.quant_fold(II,bins,ams,HH)                                   # plot reliability angles as 24-hour averages
    angis=np.zeros((100,H))
    for p in range(100):    
        angi_avg,angi_std,angi_davg,angi_navg=P4D_help.quant_fold(II,bins,angi[p],HH) 
        angis[p]=angi_avg      
    plt.clf()
    plt.imshow(angis,origin='lower',interpolation='nearest',vmin=-25.0,vmax=25.0,aspect='auto')  
    cb=plt.colorbar(orientation='vertical',aspect=40,format="%.f",ticks=range(-25,26,5))
    cb.set_label('angle'+udeg)
    cb.ax.yaxis.set_ticks_position('left')    
    for cbl in cb.ax.get_yticklabels():
        cbl.set_horizontalalignment('left')  
        cbl.set_x(-1.8)    
    ams_avg=P4D_help.sliding_median(ams_avg,window=3)        
    plt.fill_between(HX,100.0/50.0*(ams_avg-ams_std)+50.0,100.0/50.0*(ams_avg+ams_std)+50.0,alpha=0.5,color='white',lw=0)
    plt.plot(HX,100.0/50.0*ams_avg+50.0,color='white',lw=2) 
    plt.plot([0,H],[50,50],lw=2,color='white',alpha=1.0,ls='--')
    ax=P4D_help.xlabel_fold(HH,AnaSetupTimeBin,xlabel='time'+uhou,alpha=0.2) 
    plt.xlim([0,H-1])
    plt.ylim([0,100.00000001])
    plt.ylabel('plant area'+uper)
    plt.savefig(folder+'/fig_plant_fold_reliability.svg')
    np.save(folder+'/data_analyze_fold_reliability.npy',[angis,ams_avg,ams_std])   
#    plt.show()
    
###################################################### correct angles            
    
    dangle=np.arange(-3.0,3.1,1.0)*360.0                                                    # auxilliary array used to avoid 360 degree jumps in the xy-angle time series
    angles=[[] for t in TT] 
    areas=[[] for t in TT]
    times=[[] for t in TT]    
    for t in TT:                                                                            # for each track do
        for tt in table[table[:,0]==t]:
            l=int(tt[4])                                                                    # get leaf label
            i=int(tt[1])                                                                    # get image number
            q=quants[qxoc][i][l]                                                            # get origin-center xy-angle
            if(len(angles[t])>0):
                qi=np.argmin(np.abs(q+dangle-angles[t][-1]))                                # avoid 360 degree jumps in the xy-angle
            else:
                qi=3            
            angles[t].append(q+dangle[qi]) 
            areas[t].append(quants[qare][i][l])                                             # set 3D area of leaf
            times[t].append(i)                                                              # set time
    
    plt.clf()                                                                               # plot track areas and corrected xy-angles
    plt.subplot(121)
    for t in TT:
        plt.plot(times[t],angles[t],color=plt.cm.jet(1.0*t/(T-1.0)),lw=2) 
        plt.plot(times[t],np.median(angles[t])*np.ones(len(times[t])),color=plt.cm.jet(1.0*t/(T-1.0)),lw=2)      
    plt.subplot(122)
    for t in TT:
        plt.plot(times[t],areas[t],color=plt.cm.jet(1.0*t/(T-1.0)),lw=2) 
    plt.tight_layout()
    plt.savefig(folder+'/fig_analyze_aran.svg')
#    plt.show() 
    
###################################################### presort angles    
  
    im=1.0*np.vstack(itertools.imap(np.uint8,png.Reader(folder+'/focus/'+files[I-1]).read()[2]))        # read focus image for plotting  
    S=6                                                                                                 # infer order of leaf pairing from first six leaves
    SS=range(S)
    lengths=np.array([((table[:,0]==t)*(table[:,1]<400)).sum() for t in TT])                            # compute lengths of tracks restricted to first 400 images    
    if(len(lengths)<6):
        lengths=np.array([((table[:,0]==t)*(table[:,1]<9999)).sum() for t in TT])  
        print len(lengths),folder
    six=[t for t in TT if t in np.argsort(lengths)[-S:]]                                                # pick the six longest tracks    
    deg2rad=np.pi/180.0                                                                                 # auxilliary variables to convert radians and degrees
    rad2deg=180.0/np.pi
    dT=2.0*np.pi*(1.0-2.0/(1.0+np.sqrt(5.0)))*rad2deg                                                   # compute golden angle
    dayZ=-(contime[0]-15*60*24)/np.diff(contime).mean()                                                 # use day 15 to infer orderings
    beta=[np.mod(np.median(angles[s])+3600.0,360.0) for s in six]                                       # compute median xy-angle for each track
    
    plt.clf()                                                                                           # plot 3D leaf areas and linear fits
    oy,ox=data[I-1][1][0],data[I-1][1][1]                                                               # get plant origin for plotting
    plt.subplot(121)
    yax=[]
    for s in SS:        
        x=np.array(times[six[s]])
        y=np.array(areas[six[s]]) 
        if(len(x)<2 or len(y)<2):                                                                       # if track is of length one, duplicate data point
            x=np.hstack([x,x])
            y=np.hstack([y,y])
        (m,b),status=sp.optimize.curve_fit(P4D_help.fit_linear,x,y,[0.1,0.1])                           # linear fit of growth curves
        III=np.arange(dayZ,I)        
        plt.plot(x,y,ls='.',marker='o',color=plt.cm.jet(1.0*s/(S-1.0)),lw=2,mec='None',alpha=0.5)
        plt.plot(III,P4D_help.fit_linear(III,m,b),color=plt.cm.jet(1.0*s/(S-1.0)),lw=2)
        plt.text(600,y[-1],str(P4D_help.fit_linear(dayZ,m,b))+'/'+str(beta[s]),color=plt.cm.jet(1.0*s/(S-1.0)),alpha=0)
        yax.append(-P4D_help.fit_linear(dayZ,m,b))                                                      # store abscissas of linear fits
    plt.ylim(-10,55)  
    plt.subplot(122)                                                                                    # plot plant rosette and tracks
    plt.imshow(im,origin='lower',cmap='gray')
    for s in range(S):   
        track=table[table[:,0]==six[s]]
        x=track[:,2]
        y=track[:,3]        
        plt.plot(x,y,color=plt.cm.jet(1.0*s/(S-1.0)),lw=2)
        plt.axis('off')
    plt.tight_layout()    
    plt.savefig(folder+'/fig_analyze_leafit.svg')
    #plt.show()
   
###################################################### switch improvement
    
    angs=np.mod(np.array(beta)[np.argsort(yax)],360.0)                                                  # order leaf xy-angles according to magnitude of abscissas
    rng=range(S)
    damin=99999.0
    index=np.arange(S)
    direction=0
    for i0 in [-1,1]:                                                                                   # try all eight leaf orders when preserving the order of pairs
        for i1 in [-1,1]:
            for i2 in [-1,1]:
                angr=copy.deepcopy(angs)
                angr[0:2]=angs[0:2][::i0]
                angr[2:4]=angs[2:4][::i1]
                angr[4:6]=angs[4:6][::i2]   
                dA=np.arange(S)*dT                                                                      # set leaf angle spacings following golden angle                                      
                dcc=angr-angr[0]+dA                                                                     # compute distance between actual and theoretical leaf positions, counter-clockwise
                dcw=angr-angr[0]-dA                                                                     # compute distance between actual and theoretical leaf positions, clockwise
                cc=np.sum([min(np.abs(np.mod(d,360.0)),np.abs(np.mod(-d,360.0))) for d in dcc])         # compute sum of absolute angle differences
                cw=np.sum([min(np.abs(np.mod(d,360.0)),np.abs(np.mod(-d,360.0))) for d in dcw])
                cc=np.min(np.vstack([np.mod(np.diff(angr)+dT,360.),np.mod(-np.diff(angr)-dT,360.)]),0).mean()
                cw=np.min(np.vstack([np.mod(np.diff(angr)-dT,360.),np.mod(-np.diff(angr)+dT,360.)]),0).mean()
                if(cc<cw):
                    direction=-1
                else:
                    direction=1
                da=min(cc,cw)
                #print i0,i1,i2,da
                if(da<damin):                                                                           # store ordering of leaves of sum of absolute angles differences is minimal 
                    damin=da
                    diri=direction
                    index[0:2]=rng[0:2][::i0]
                    index[2:4]=rng[2:4][::i1]
                    index[4:6]=rng[4:6][::i2]
    angz=angs[np.array(index)]                                                                          # order angles according to optimal matching with golden angle phyllotaxis

############################################################################### overlap
           
    lengtha=np.array([(table[:,0]==t).sum() for t in TT])                               # compute lengths of tracks
    alphi=np.array([t for t in TT if (table[:,0]==t).sum()>AnaTrackMinlen])             # take indices of tracks that exceed a certain length
    A=len(alphi)        
    alpha=np.mod([np.median(angles[a]) for a in alphi],360.0)                           # compute xy-angles of sufficiently long tracks
    overlap=np.zeros((A,A))                                                             # auxilliary array to store overlaps
    ao=[]    
    for t in alphi:                                                                     # set frames covered by each track 
        track=table[table[:,0]==t][:,1]
        ao.append(set(track))
    for a1 in range(A):                                                                 # for all pairs of tracks  
        for a2 in range(A): 
            overlap[a1,a2]=len(ao[a1].intersection(ao[a2]))                             # compute number of frames in both tracks
    
    plt.clf()                                                                           # plot overlap matrix
    plt.imshow(overlap,interpolation='nearest',cmap='gray')
    plt.tight_layout()
    plt.savefig(folder+'/fig_analyze_overlap.svg') 
#    plt.show()  
  
###################################################### assign angles  
            
    ### assign early leafs first         
    bet=np.mod(np.hstack([angz[0:-1],angz[-1]+diri*np.arange(B-5)*dT*1.0]),360.0)               # set hypothetical leaf angles and continue with golden angle for up to eight leaves
    order=np.arange(0,B)#np.hstack([np.arange(2,S),np.arange(0,2),np.arange(S,B)])                             # assign leaves 1 to 4 first, then the hypocotyls, then leaves 5-6
    xx=np.zeros((A,B))                                                                          # create auxilliary array for assignment of track a to leaf position b
    for bi,b in enumerate(bet[order]):                                                          # for each leaf angle do
        bio=order[bi]                                                                           # get leaf position index
        dang=np.mod(np.abs(alpha-b),360.0)                                                      # compute absolute difference in track and leaf angles
        aa=np.argsort(dang)                                                                     # sort angle differences
        al=np.argsort(lengtha[aa[:6]])[::-1]                                                    # sort tracks with the six smallest angle deviations by length
        for ai in aa[:6][al]:                                                                   # for each of those tracks do
            xi=np.where(xx[:,bio])[0]                                                           # get overlaps of tracks assigned to current leaf positions  
            if(np.sum(overlap[xi],0)[ai]<AnaAssignMaxOverlap and dang[ai]<AnaAssignMaxDAngles): # if new track does not strongly overlap with previously assigned ones and the angle deviation is small enough
                if(xx[ai,:].sum()==0):                                                          # if the track has not been assigned before
                    xx[ai,bio]=1                                                                # assign the track to the current leaf position
    
    plt.clf()                                                                                   # plot track to leaf assignments 
    plt.imshow(xx,interpolation='nearest',cmap='gray')
    plt.tight_layout()
    plt.savefig(folder+'/fig_analyze_optimal.svg')  
    np.save(folder+'/data_analyze_assignment.npy',[[S,B,A],bet,alpha,alphi,xx])         
#    plt.show()     

####################################################### plot angles

    plt.clf()                                                                                                                       # plot focus image and tracks according to infered leaf assignments
    plt.imshow(im,origin='lower',cmap='gray')
    for b in range(B):                                                                                                              # plot hypothetical leaf positions
        plt.plot([ox,400*np.sin(bet[b]*deg2rad)+ox],[oy,400*np.cos(bet[b]*deg2rad)+oy],color=cobs[b],lw=2,ls='--')  
        plt.text(420*np.sin(bet[b]*deg2rad)+ox,420*np.cos(bet[b]*deg2rad)+oy,leafs[b],color='white',ha='center',va='center') 
    for s in range(S):                                                                                                              # plot cytoledons and first four leaf positions
        col=list(np.argsort(yax)[index]).index(s)
        track=table[table[:,0]==six[s]]
        x=track[:,2]
        y=track[:,3]  
        plt.plot(x,y,color=plt.cm.jet(1.0*col/(B-1.0)),lw=2,alpha=1.0)        
    for ai,a in enumerate(alphi):                                                                                                   # plot tracks with colors matching the leaf assignments
        track=table[table[:,0]==a]
        x=track[:,2]
        y=track[:,3]  
        xi=np.where(xx[ai]>0.1)[0]
        if(len(xi)>0):        
            col=xi[0]
            plt.plot(x,y,color=plt.cm.jet(1.0*col/(B-1.0)),lw=2,ls=':')
        else:
            plt.plot(x,y,color='white',lw=2,alpha=0.5,ls=':')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(folder+'/fig_analyze_assignment.svg')
#    plt.show()       

###################################################### write spreadsheet full
    
    QA=Q0+B*Q1                                                                                                  # total number of computed featues
    full=np.zeros((I,QA))*np.nan                                                                                # create auxilliary table for full data
    name_full=np.hstack(names[:Q0]+[[str(b+1).zfill(2)+':'+names[q] for q in QQ[Q0:]] for b in range(B)])       # set column titles  
    for i in range(I):                                                                                          # for each image do
        for q in range(Q0):                                                                                     
            full[i,q]=quants[q][i][0]                                                                           # write all plant features
        for b in range(B):                                                                                      # for each leaf do
            ai=np.where(xx[:,b]>0.1)[0]                                                                         # get all tracks assigned to the current leaf
            for q in range(Q1): 
                for a in alphi[ai]:                                                                             # for each assigned track do
                    track=table[table[:,0]==a]
                    if(i in track[:,1]):                                                                        # if the current track covers the current frame
                        l=int(track[track[:,1]==i][0,4])                                                        # get the original leaf number    
                        full[i,Q0+b*Q1+q]=quants[Q0+q][i][l]                                                    # write leaf feature
                        
    for q in range(QA):                                                                                         # correct xy-angles to avoid 360 degree jumps
        if('xy_' in name_full[q] or 'z_' in name_full[q]):            
            full[:,q]=P4D_help.angle_nojump(full[:,q])

    a=np.vstack([name_full,full])                                                                               # stack column titles and data  
    a[a=='nan']=''
    ff=open(folder+'/data_spreadsheet_full.csv',"wb")                                                           # save full data
    writer=csv.writer(ff,lineterminator = '\n')       
    writer.writerows(a)    
    ff.close()
       
###################################################### write spreadfold fold and hist
        
    QT=2*Q0+2*B*Q1                                                                                              # total number of computed features after 24-hour averaging
    fold=np.zeros((H,QT))*np.nan                                                                                # create auxilliary table for folded data
    hist=np.zeros((H,QT))*np.nan                                                                                # create auxilliary table for histogram data
    name_fold=np.hstack([[n+app for app in ['_avg','_std']] for n in name_full])                                # set column titles  
    for q in range(QA):                                                                                         # for each feature
        props=full[:,q]                                                                                         # get full data
        rer='_2Darea' in name_full[q] or '_3Darea' in name_full[q]                                              
        if(rer):                                                                                                # if 2D or 3D areas are considered, compute RER as feature of interest
            ID,props_dff,ms_davg,ms_navg=P4D_help.quant_hist(props,I,AnaSetupDRER,AnaSetupTimeBin,darks,contime)
        else:                                                                                                   # otherwise, do not modify time series
            ID,props_dff=II,props
        props_avg,props_std,props_davg,props_navg=P4D_help.quant_fold(ID,bins,props_dff,HH)                     # fold data to obtain 24-hour average and histogram for changes at day and night
        fold[:H,2*q+0]=props_avg                                                                                # set averages of folded data 
        fold[:H,2*q+1]=props_std                                                                                # set standard deviation of folded data 
        if(rer):
            R=len(ms_davg)
            hist[:R,2*q+0]=ms_davg                                                                              # set histogram results for the day
            R=len(ms_navg)
            hist[:R,2*q+1]=ms_navg                                                                              # set histogram results for the night
        else:
            R=len(props_davg)
            hist[:R,2*q+0]=props_davg
            R=len(props_navg)
            hist[:R,2*q+1]=props_navg
    
    a=np.vstack([name_fold,fold])                                                                               # stack column titles and data  
    a[a=='nan']=''    
    ff=open(folder+'/data_spreadsheet_fold.csv',"wb")                                                           # save fold data
    writer=csv.writer(ff,lineterminator = '\n')       
    writer.writerows(a)        
    ff.close()    
    a=np.vstack([name_fold,hist])                                                                               # stack column titles and data  
    a[a=='nan']=''    
    ff=open(folder+'/data_spreadsheet_hist.csv',"wb")                                                           # save hist data
    writer=csv.writer(ff,lineterminator = '\n')       
    writer.writerows(a)        
    ff.close()
    
################################################################# leaf appearances

    plt.clf()                                                                                                   # compute and plot timing of leaf appearances
    rs=[]
    for b in range(B):                                                                                        # for each leaf do
        areay=full[:,qare+b*Q1]                                                                                 # get 3D areas
        idx=(areay==areay)                                                                                      # pick only non-nan values  
        if(idx.sum()>2):                                                                                        # if time series has more than two entries 
            J=len(areay[idx])
            sigma=[np.max(np.abs(areay[idx][i]-areay[idx][P4D_help.bounds(i-10,0,J):P4D_help.bounds(i+10,0,J)])) for i in range(J)]
            if(b<4):
                poly,status=sp.optimize.curve_fit(P4D_help.fit_linear,II[idx],areay[idx],sigma=sigma)           # fit increasing first-order polynomial for older leaves and cotyledons
            else:
                poly,status=sp.optimize.curve_fit(P4D_help.fit_parabola,II[idx],areay[idx],sigma=sigma)         # fit convex second-order polynomial for younger leaves
            poly[0]=np.abs(poly[0])                                                                             # force polynomial to be convex    
            roots=np.roots(poly).real                                                                           # compute real values of zeros
            r=max(roots)                                                                                        # pick largest zero as timing of leaf appearance
            IJ=np.arange(min(10000-1,r),10000)                                                                  # auxilliary array for plotting
            areaz=np.poly1d(poly)(IJ)                                                                           # areas from polynomial fit         
            plt.errorbar(II[idx],areay[idx],sigma,color=cobs[b],alpha=0.1)
            plt.plot(II[idx],areay[idx],color=cobs[b],ls='.',marker='o',mec='none',lw=2,alpha=0.1)              # plot leaf areas
            plt.plot(IJ,areaz,color=cobs[b],lw=2,alpha=1.0,ls='-')                                              # plot polynomial fit
            plt.plot([r,r],[-100,1000],color=cobs[b],lw=2,ls='--',alpha=1.0)                                    # plot timing of leaf appearance
            plt.text(r,10,leafs[b],ha='center')                                                                          # label leaf by number
            rs.append(r)                                                                                        # store leaf appearance
        else:
            rs.append(np.nan)
    polt=np.polyfit(II,contime,1)                                                                               # linear fit of image times in minutes
    rd=np.poly1d(polt)(rs)/(24.0*60.0)                                                                          # compute timing of leaf appearance in days
    plt.plot([-9999,9999],[0,0],color='black',lw=2,ls='--')    
    ld=24.0*60.0/np.diff(contime).mean()                                                                        # auxilliary variables to convert frame numbers into days
    di=np.argmax(np.diff(days))+1
    do=days[di]
    lt=di+np.arange(-60-do%2,20)*ld
    ll=(do+np.arange(-60-do%2,20)).astype('int')
    plt.xticks(lt[::2],ll[::2])
    plt.axis([-400,900,-10,40])
    plt.xlabel('time'+udas)
    plt.ylabel('leaf areas'+ummm)
    plt.tight_layout() 
    for b in range(B/2):                                                                                        # pairwise sorting of leaf appearances
        if(np.isnan(rd[2*b+0:2*b+2]).sum()==0):
            rd[2*b+0:2*b+2]=np.sort(rd[2*b+0:2*b+2])
    rd=np.sort(rd)
    np.save(folder+'/data_analyze_appearances.npy',rd)                                                          # save timing of leaf appearance
    plt.savefig(folder+'/fig_analyze_appearances.svg')                                                          # plot timing of leaf appearance
#    plt.show()
#    print rd
    
        
    S=F                                                                                                                             # compute number of folders to process
    s=f                                                                                                                             # compute number of current image
    dt=time.time()-tempo                                                                                                            # compute duration of current analysis step
    print 'analyze:','folder',f+1,F,'time spent',dt,'seconds','time remaining',(S-s)*dt/(3600.0*MainCores),'hours'                  # print computation time statistics
        
    return 0
    
    

    


        
    

        
