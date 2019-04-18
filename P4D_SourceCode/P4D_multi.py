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
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np
import pandas
import scipy as sp
import scipy.cluster
import scipy.misc
import scipy.ndimage
import scipy.optimize
import scipy.spatial
import scipy.stats
import time

import P4D_help                                 # implementations of various auxiliary functions
reload(P4D_help)

########################################################## multi

def multi(inp):
    '''Function generates a series of customized plots for publication. Takes feature tables as inputs.'''

    project,folders,geno,G,MainPath,MainCores,AnaTrackStop,AnaTrackRange,AnaTrackMemory,AnaTrackMinlen,AnaAssignMaxDAngles,AnaAssignMaxOverlap,AnaSetupDRER,AnaSetupNightStart,AnaSetupTimeBin,DoTrack,temp=inp     # input parameters
    tempo=time.time()                                                                                                                                                                                       # remember starting time
    GG=range(G)                                                                                                                                                                                             # list of genotypes
    folderz=[[] for g in GG]            
    genon=sorted(list(set(geno)))                                                                                                                                                                    # auxilliary list for folders of one genotype
    for gi,g in enumerate(genon):                                                                                                                                                         
        folderz[gi]=np.array(folders)[geno==g]
    quants,names,units,[HX,HH,Q0,Q1,lx,ly,dxy]=np.load(folders[0]+'/data_analyze_quants.npy')                                                                                                               # load tracking and plotting parameters
    [ummm,ucmx,upph,udas,uhou,ummx,udeg,uper,uxxx],[qare,qcirc,qcomp,qcurv,qasra,qxoc,qxbt,qzoc,qzbt,qzlr,qpox,qpoy,qpoz]=np.load(folders[0]+'/data_analyze_units.npy')                                     # load unit labels and feature numbers
    B=8                                                                                                                                                                                                     # set maximal number of leaves
    mpl.rcParams.update({'font.size':18})                                                                                                                                                                   # set plotting font size
    cfull=pandas.read_csv(folders[0]+'/data_spreadsheet_full.csv',encoding='utf-8',delimiter=',')                                                                                                           # load full time series data
    full=cfull.as_matrix()  
    J,QA=full.shape         
    I=AnaTrackStop#  J-10#                                                                                                                                                                                  # set number of images to include in analysis
    II=np.arange(I)  
    Fg=[len(folderz[g]) for g in GG]                                                                                                                                                                        # compute number of folders per genotype
    prer3D=[[np.zeros(I) for f in range(Fg[g])] for g in GG]                                                                                                                                                # auxilliary lists for plant features
    prer2D=copy.deepcopy(prer3D)    
    pu3D=copy.deepcopy(prer3D)    
    pu2D=copy.deepcopy(prer3D)    
    pdavg=copy.deepcopy(prer3D)    
    pnavg=copy.deepcopy(prer3D)   
    prelc=copy.deepcopy(prer3D)
    prela=copy.deepcopy(prer3D)
    pfelc=copy.deepcopy(prer3D)
    pfela=copy.deepcopy(prer3D)
    pucomp=copy.deepcopy(prer3D)
    pucirc=copy.deepcopy(prer3D)
    puasym=copy.deepcopy(prer3D)
    lrers=[[[np.zeros(I) for f in range(Fg[g])] for b in range(B)] for g in GG]                                                                                                                             # auxilliary lists for leaf features
    luarea=copy.deepcopy(lrers)
    luasra=copy.deepcopy(lrers)
    ldavg=copy.deepcopy(lrers)
    lnavg=copy.deepcopy(lrers)
    lozbt=copy.deepcopy(lrers)
    lozoc=copy.deepcopy(lrers)    
    lozob=copy.deepcopy(lrers)    
    lozot=copy.deepcopy(lrers)     
    loxoc=copy.deepcopy(lrers)
    loxbt=copy.deepcopy(lrers)
    locurv=copy.deepcopy(lrers)
    lucurv=copy.deepcopy(lrers)
    luzbt=copy.deepcopy(lrers)
    luxoc=copy.deepcopy(lrers)
    luxot=copy.deepcopy(lrers)
    lwzbt=copy.deepcopy(lrers)
    lwzob=copy.deepcopy(lrers)
    lwzot=copy.deepcopy(lrers)
    lucirc=copy.deepcopy(lrers)
    lucomp=copy.deepcopy(lrers)
    ltime=copy.deepcopy(lrers)    
    gc=np.repeat(np.linspace(0,1,G),2)
    gd=np.array([-0.1*s for s in np.sign(np.arange(G)-(G-1)/2.)])           
    gc[::2]+=gd
    colors=[plt.cm.jet(g) for g in gc]                                                                                                                                  # list of genotype colors               
    cobs=[plt.cm.jet(1.0*b/(B-1.0)) for b in range(8)]                                                                                                                  # list of leaf colors
    stylo=['solid','dashed','solid','dashed']                                                                                                                           # list of bar plot styles
    styles=['-','--','-.',':']                                                                                                                                          # list of line styles
    dn=['day','night','day','night']                                                                                                                                    # list of day-night labels
    
    ########################################### loading
    
    print 'multi:','gene',0,G,'folder',0,0,'loading'
    for gi in GG:                                                                                                                                                       # for each genotype do
        for fi,folder in enumerate(folderz[gi]):                                                                                                                        # for each folder of current genotype do
            print 'multi:','geno',gi+1,G,'folder',fi+1,Fg[gi],'loading',folder  
            cfull=pandas.read_csv(folder+'/data_spreadsheet_full.csv',encoding='utf-8',delimiter=',')                                                                   # load full time series data
            full=cfull.as_matrix()  
            J,QA=full.shape            
            cfold=pandas.read_csv(folder+'/data_spreadsheet_fold.csv',encoding='utf-8',delimiter=',')                                                                   # load folded time series data
            fold=cfold.as_matrix()
            H,QT=fold.shape
            HX=np.arange(H)  
            chist=pandas.read_csv(folder+'/data_spreadsheet_hist.csv',encoding='utf-8',delimiter=',')                                                                   # load histogram data
            hist=chist.as_matrix() 
            angi,ams=np.load(folder+'/data_analyze_full_reliability.npy')                                                                                       # load full reliability angle data
            angis,ams_avg,ams_std=np.load(folder+'/data_analyze_fold_reliability.npy')                                                                                  # load folded reliability angle data
            timing=np.load(folder+'/data_analyze_appearances.npy')                                                                                                      # load timings of leaf appearances
            prelc[gi][fi]=angis                                                                                                                                         # append plant features
            prela[gi][fi]=ams_avg               
            pfelc[gi][fi]=np.zeros((100,I))
            pfelc[gi][fi][:,:I]=angi[:,:I]  
            pfela[gi][fi][:I]=ams[:I]   
            pu2D[gi][fi][:I]=full[:,8][:I] 
            pu3D[gi][fi][:I]=full[:,9][:I] 
            prer2D[gi][fi]=fold[:,8*2]
            prer3D[gi][fi]=fold[:,9*2]            
            pdavg[gi][fi]=hist[:,9*2+0]
            pnavg[gi][fi]=hist[:,9*2+1]  
            pucirc[gi][fi][:I]=full[:,11][:I] 
            pucomp[gi][fi][:I]=full[:,12][:I] 
            puasym[gi][fi][:I]=full[:,13][:I]               
            for bi in range(B):                                                                                                                                         # append leaf features
                ldavg[gi][bi][fi]=hist[:,2*24+2*bi*Q1+0]
                lnavg[gi][bi][fi]=hist[:,2*24+2*bi*Q1+1]              
                lrers[gi][bi][fi]=fold[:,2*qare+2*bi*Q1]                
                lozbt[gi][bi][fi]=fold[:,2*qzbt+2*bi*Q1]
                lozoc[gi][bi][fi]=fold[:,2*qzoc+2*bi*Q1]
                lozob[gi][bi][fi]=fold[:,2*39+1+2*bi*Q1]
                lozot[gi][bi][fi]=fold[:,2*40+2*bi*Q1]
                loxoc[gi][bi][fi]=fold[:,2*qxoc+2*bi*Q1]
                loxbt[gi][bi][fi]=fold[:,2*qxbt+2*bi*Q1]
                locurv[gi][bi][fi]=fold[:,2*qcurv+2*bi*Q1]   
                luzbt[gi][bi][fi][:I]=full[:,qzbt+bi*Q1][:I] 
                luxoc[gi][bi][fi][:I]=full[:,qxoc+bi*Q1][:I] 
                luxot[gi][bi][fi][:I]=full[:,35+bi*Q1][:I] 
                luarea[gi][bi][fi][:I]=full[:,qare+bi*Q1][:I] 
                luasra[gi][bi][fi][:I]=full[:,qasra+bi*Q1][:I] 
                lucurv[gi][bi][fi][:I]=full[:,qcurv+bi*Q1][:I]               
                lucirc[gi][bi][fi][:I]=full[:,qcirc+bi*Q1][:I] 
                lucomp[gi][bi][fi][:I]=full[:,qcomp+bi*Q1][:I] 
                lwzbt[gi][bi][fi][:I]=P4D_help.fosci(full[:,41+bi*Q1])[:I]
                lwzob[gi][bi][fi][:I]=P4D_help.fosci(full[:,39+bi*Q1])[:I]              
                lwzot[gi][bi][fi][:I]=P4D_help.fosci(full[:,40+bi*Q1])[:I] 
                ltime[gi][bi][fi]=timing[bi]       
    days=full[:,0][:I]                                                                              # set day info of images
    hours=full[:,1][:I]                                                                             # set hour info of images        
    darks=full[:,2][:I]                                                                             # set dark info of images    
    midnights=full[:,3][:I]                                                                         # set midnight info of images    
    bins=full[:,4].astype('int')[:I]                                                                # set bin info of images    
    contime=full[:,5][:I]                                                                           # set minute info of images    
    dh=np.mean(np.diff(contime))                                                                    # set average time step in minutes

    #sys.exit()
                
################################################# paper 4a: plant full areas

    aa=[[] for g in GG]
    idx=[[] for g in GG]
    for g in GG:
        aa[g]=np.array([pu3D[g][f][0] for f in range(Fg[g])])                                   # create list of folders with initial areas within 2mm^2 of median
        idx[g]=np.abs(aa[g]-np.median(aa[g]))<99992.0
        idx[g]=np.where(idx[g])[0]
        
    plt.clf()                                                                                   # plot plant growth curves with mean +- standard deviation    
    ax=plt.subplot(111)     
    window=20
    for g in GG:
        p2d=[pu2D[g][f] for f in idx[g]]                                                        # get 2D areas
        p3d=[pu3D[g][f] for f in idx[g]]                                                        # get 3D areas
        savg=P4D_help.sliding_median(sp.stats.nanmean(p2d,0),window=window)
        aavg=P4D_help.sliding_median(sp.stats.nanmean(p3d,0),window=window)   
        sstd=P4D_help.sliding_median(sp.stats.nanstd(p2d,0,bias=1),window=window)
        astd=P4D_help.sliding_median(sp.stats.nanstd(p3d,0,bias=1),window=window)   
        #ax.fill_between(II,savg-sstd,savg+sstd,color=colors[2*g+1],lw=0,alpha=0.2)
        ax.fill_between(II,aavg-astd,aavg+astd,color=colors[2*g+0],lw=0,alpha=0.2) 
        #ax.plot(II,savg,color=colors[2*g+1],lw=2,ls=styles[1])
        ax.plot(II,aavg,color=colors[2*g+0],lw=2,ls=styles[0]) 
    P4D_help.xlabel_full(II,darks,midnights,days,xlabel='time'+udas,alpha=0.2)                  # indicate nights by gray overlays
    plt.ylabel('plant 3D area'+ummm)
    plt.xlim([0,900])
    plt.ylim([0,500])
    axins=mpl_toolkits.axes_grid1.inset_locator.zoomed_inset_axes(ax,2.5,loc=2)                 # plot enlarged section of the growth curve for one plant
    for g in GG:        
        p2d=[pu2D[g][f] for f in idx[g]]                                                        # get 2D areas
        p3d=[pu3D[g][f] for f in idx[g]]                                                        # get 3D areas
        savg=P4D_help.sliding_median(sp.stats.nanmean(p2d,0),window=window)
        aavg=P4D_help.sliding_median(sp.stats.nanmean(p3d,0),window=window)   
        sstd=P4D_help.sliding_median(sp.stats.nanstd(p2d,0,bias=1),window=window)
        astd=P4D_help.sliding_median(sp.stats.nanstd(p3d,0,bias=1),window=window)   
        #axins.fill_between(II,savg-sstd,savg+sstd,color=colors[2*g+1],lw=0,alpha=0.2)
        axins.fill_between(II,aavg-astd,aavg+astd,color=colors[2*g+0],lw=0,alpha=0.2) 
        #axins.plot(II,savg,color=colors[2*g+1],lw=2,ls=styles[1])
        axins.plot(II,aavg,color=colors[2*g+0],lw=2,ls=styles[0]) 
    P4D_help.xlabel_full(II,darks,midnights,days,xlabel='',alpha=0.2)
    x0,x1,y0,y1=250,450,10,130 
    axins.set_xlim(x0,x1)
    axins.set_ylim(y0,y1)    
    plt.xticks(visible=False)
    plt.yticks(visible=False)    
    mpl_toolkits.axes_grid1.inset_locator.mark_inset(ax,axins,loc1=3,loc2=4,fc="none",ec='gray',lw=2)  
    ax2=ax.twinx()
    ax2.set_ylim([0,500])
    plt.ylabel('plant 2D area'+ummm)
    plt.savefig(project+'/fig4a_plantareas.svg')                                                # save growth curve plot
    #plt.show()      

    
################################################################ paper 4b: plant fold rer
    
    plt.clf()                                                                                           # plot plant relative expansion rates (RERs) for plants of similar areas, see above
    di=np.argmax((contime-contime[0])>AnaSetupDRER*60.0)                                                # compute number of frames to bridge the desired time step for the RER computation
    HC=range(H)     
    HC=range(H)[-di:]+range(H)[:-di]                                                                    # shift the RER plots by this frame number    
    HC=HC
    for g in GG:
        p2D=[prer2D[g][f] for f in idx[g]]                                                              # get 2D RERs
        avg2D=P4D_help.sliding_median(sp.stats.nanmean(p2D,0),window=2)[HC]
        std2D=P4D_help.sliding_median(sp.stats.nanstd(p2D,0,bias=1),window=2)[HC]  
        p3D=[prer3D[g][f] for f in idx[g]]                                                              # get 3D RERs
        avg3D=P4D_help.sliding_median(sp.stats.nanmean(p3D,0),window=2)[HC]
        std3D=P4D_help.sliding_median(sp.stats.nanstd(p3D,0,bias=1),window=2)[HC]        
        #plt.fill_between(HX,avg2D-std2D,avg2D+std2D,color=colors[2*g+1],lw=0,alpha=0.2)
        #plt.plot(HX,avg2D,color=colors[2*g+1],alpha=1.0,lw=2,ls='--')
        plt.fill_between(HX,avg3D-std3D,avg3D+std3D,color=colors[2*g+0],lw=0,alpha=0.2)
        plt.plot(HX,avg3D,color=colors[2*g+0],alpha=1.0,lw=2,ls='-',label=genon[g])
    ax=P4D_help.xlabel_fold(HH,AnaSetupTimeBin,xlabel='time'+uhou)                                      # indicate nights by gray overlays
    plt.legend()
    plt.ylabel('plant RER'+upph)
    plt.ylim(-0.035,0.045)#ylim(-0.025,0.035)
    plt.savefig(project+'/fig4b_plantrer.svg')
    #plt.show() 
    
################################################################ paper 4x: plant fold rer extended
  
    plt.clf()                                                                                           # plot plant relative expansion rates (RERs) for plants of similar areas, see above
    di=np.argmax((contime-contime[0])>AnaSetupDRER*60.0)                                                # compute number of frames to bridge the desired time step for the RER computation
    HC=range(H)[-di:]+range(H)[:-di]                                                                    # shift the RER plots by this frame number    
    for g in GG:
        p3D=[prer3D[g][f] for f in idx[g]]                                                              # get 3D RERs
        avg3D=P4D_help.sliding_median(sp.stats.nanmean(p3D,0),window=2)[HC]
        std3D=P4D_help.sliding_median(sp.stats.nanstd(p3D,0,bias=1),window=2)[HC]        
        avg3D=np.hstack([avg3D,avg3D[:12]])
        std3D=np.hstack([std3D,std3D[:12]])
        hx=np.hstack([HX,range(144,156)])
        plt.fill_between(hx,avg3D-std3D,avg3D+std3D,color=colors[2*g+0],lw=0,alpha=0.2)
        plt.plot(hx,avg3D,color=colors[2*g+0],alpha=1.0,lw=2,ls='-',label=genon[g])
    ax=P4D_help.xlabel_fold(HH,AnaSetupTimeBin,xlabel='time'+uhou)                                      # indicate nights by gray overlays
    plt.ylabel('plant RER'+upph)
    plt.ylim(-0.005,0.035)#ylim(-0.025,0.035)
    plt.yticks(np.arange(-0.005,0.036,0.005),np.arange(-0.5,3.6,0.5))
    plt.xlim(0,155)
    plt.savefig(project+'/fig4x_plantrerex.svg')
    #plt.show()     
    
################################################################ paper 4y: plant rer vs angle
  
    plt.clf()                                                                                         
    di=np.argmax((contime-contime[0])>AnaSetupDRER*60.0)                                             
    HC=range(H)[-di:]+range(H)[:-di]                                                                   
    for g in GG:
        plt.clf() 
        p3D=[prer3D[g][f] for f in idx[g]]  
        ang=[prela[g][f] for f in idx[g]] 
        amu=sp.stats.nanmean(ang,0)
        avgan=P4D_help.sliding_median(np.diff(np.hstack([amu,amu[0]])),window=2)        
        avg3D=P4D_help.sliding_median(sp.stats.nanmean(p3D,0),window=2)[HC]   
        ax1=plt.subplot(111)
        plt.plot(avg3D,color=colors[2*g+0],alpha=1.0,lw=2,ls='-',label=genon[g])
        plt.ylabel('plant RER'+upph)
        plt.xlabel('time'+uhou)
        plt.ylim(-0.005,0.035)#ylim(-0.025,0.035)
        plt.yticks(np.arange(-0.005,0.036,0.005),np.arange(-0.5,3.6,0.5))
        ax2=ax1.twinx()
        plt.plot(avgan/5.0,color=colors[2*g+0],alpha=1.0,lw=2,ls='--',label=genon[g])
        plt.ylabel('change in hyponasty angle [$^{\circ}/h$]')
        plt.ylim(-0.15,0.1)
        ax=P4D_help.xlabel_fold(HH,AnaSetupTimeBin,xlabel='time'+uhou)       
        plt.xlim(0,143)        
        plt.savefig(project+'/fig4y_plantreran'+str(g)+'.svg')          
        plt.clf()
        aa=len(avgan)
        bb=np.hstack([avg3D,avg3D])
        cc=np.correlate(avgan,bb)             
        dd=np.array([sp.stats.pearsonr(avgan,bb[1*aa-i:2*aa-i]) for i in range(aa)])
        ax1=plt.subplot(111)   
        plt.plot(cc,color=colors[2*g+0],alpha=1.0,lw=2,ls='-',label=genon[g])
        plt.ylim(-0.06,0.06)
        P4D_help.peak(cc,1.0/6.0,uhou,colors[2*g+0])
        ax2=ax1.twinx()
        plt.plot(dd[:,1],lw=2,color=colors[2*g+0],ls=':')#plt.yscale('log')
        plt.plot(0*dd[:,1]+0.05,lw=2,color='black',ls='--')
        plt.xticks((np.arange(24.0)*60.0/AnaSetupTimeBin).astype('int')[::4],np.divide(HX[::6],60.0/AnaSetupTimeBin).astype('int')[::4])   
        plt.xlim(0,143)        
        plt.savefig(project+'/fig4y_plantreran'+str(g)+'cc.svg')               
        #plt.show()        
        
    ################################################################ paper 4c: plant fold and hist
        
    plt.clf()                                                                                                   # plot plant RERs at day and night and test for differences
    tts=[[] for g in GG]
    for g in GG:
        pd=np.hstack([pdavg[g][f][:5] for f in idx[g]])
        pn=np.hstack([pnavg[g][f][:5] for f in idx[g]])
        pd=pd[pd==pd]
        pn=pn[pn==pn]
        davg=sp.stats.nanmean(pd)
        dstd=sp.stats.nanstd(pd,bias=1)
        navg=sp.stats.nanmean(pn)
        nstd=sp.stats.nanstd(pn,bias=1)        
        plt.errorbar(2*g+1,davg,yerr=dstd,ls='.',ecolor=colors[2*g+1],elinewidth=4,capsize=10,capthick=4,zorder=1)        
        plt.bar(2*g+1,davg+99,width=0.9,lw=4,align='center',edgecolor=colors[2*g+1],color='white',zorder=2,ls='solid',bottom=-99)    
        plt.errorbar(2*g+2,navg,yerr=nstd,ls='.',ecolor=colors[2*g+0],elinewidth=4,capsize=10,capthick=4,zorder=1)        
        plt.bar(2*g+2,navg+99,width=0.9,lw=4,align='center',edgecolor=colors[2*g+0],color='white',zorder=2,ls='solid',bottom=-99)  
        tts[g]=sp.stats.ttest_ind(pd,pn)[1]
    for g in GG:
        plt.text(2*g+1.5,0.025*0.9,'p='+'%.1e'%tts[g],ha='center')         
    plt.ylim(ymin=0,ymax=0.025)
    plt.xlim([0,1+2*G])    
    plt.xticks(range(1,1+2*G),dn*G)
    plt.ylabel('plant RER'+upph)    
    plt.savefig(project+'/fig4c_planthist.svg')
    #plt.show()      
    
    ################################################################ paper 4d: plant full reli      
    
    plt.clf()                                                                                                   # plot reliability angles over the full time
    reli=sp.stats.nanmean(pfelc[0],0) 
    plt.imshow(reli,origin='lower',interpolation='nearest',vmin=-25.0,vmax=25.0,aspect='auto')  
    cb=plt.colorbar(orientation='vertical',aspect=40,format="%.f",ticks=range(-25,26,5))
    cb.set_label('leaf z-angle base-tip'+udeg)      
    for g in range(G):
        avg=P4D_help.sliding_median(sp.stats.nanmean(pfela[g],0),window=2)*2.0+50.0
        std=P4D_help.sliding_median(sp.stats.nanstd(pfela[g],0,bias=1),window=2)
        plt.fill_between(II,avg-std,avg+std,color=colors[2*g+0],lw=0,alpha=0.2)
        plt.plot(II,avg,color=colors[2*g+0],alpha=1.0,lw=2)
    P4D_help.xlabel_full(II,darks,midnights,days,xlabel='time'+udas,alpha=0.2)
    plt.ylabel('plant area'+uper)
    plt.xlabel('time'+udas)
    plt.xlim([0,900])
    plt.ylim([0,100.00000001])
    plt.savefig(project+'/fig4d_plantfullreli.svg')
    #plt.show()   

    plt.close('all') 
    
    ################################################################ paper 4e: plant fold reli          
    
    plt.clf()                                                                                                   # plot reliability angles as 24-hour averages
    reli=sp.stats.nanmean(prelc[0],0)  
    plt.imshow(reli,origin='lower',interpolation='nearest',vmin=-25.0,vmax=25.0,aspect='auto')  
    cb=plt.colorbar(orientation='vertical',aspect=40,format="%.f",ticks=range(-25,26,5))
    cb.set_label('leaf z-angle base-tip'+udeg)      
    for g in range(G):
        avg=P4D_help.sliding_median(sp.stats.nanmean(prela[g],0),window=2)*2.0+50.0
        std=P4D_help.sliding_median(sp.stats.nanstd(prela[g],0,bias=1),window=2)*np.sqrt(2.0)
        plt.fill_between(HX,(avg-std)[HX],(avg+std)[HX],color=colors[2*g+1],alpha=0.6,lw=0)
        plt.plot(HX,avg[HX],color=colors[2*g+0],alpha=1.0,lw=2)
    ax=P4D_help.xlabel_fold(HH,AnaSetupTimeBin,xlabel='time'+uhou) 
    plt.ylabel('plant area'+uper)
    plt.xlim([0,H-1])
    plt.ylim([0,100.00000001])
    plt.savefig(project+'/fig4e_plantfoldreli.svg')
    #plt.show()       
    
    ##################################################################### paper 4f: plant full asym comp
    
    plt.clf()                                                                                                   # plot plant asymmetry and compactness over time
    ax1=plt.subplot(111)
    for g in range(G):
        avg=P4D_help.sliding_median(sp.stats.nanmean(puasym[g],0),window=20)
        std=P4D_help.sliding_median(sp.stats.nanstd(puasym[g],0,bias=1),window=20)/1.0
        ax1.fill_between(II,avg-std,avg+std,color=colors[2*g+0],alpha=0.2,lw=0)
        ax1.plot(II,avg,color=colors[2*g+0],lw=2,alpha=1.0)    
    plt.ylabel('plant asymmetry'+ummx)   
    plt.ylim(ymin=0,ymax=1.0)
    plt.xlabel('time'+udas)   
    ax2=ax1.twinx()
    for g in range(G):
        avg=P4D_help.sliding_median(sp.stats.nanmean(pucomp[g],0),window=2)
        std=P4D_help.sliding_median(sp.stats.nanstd(pucomp[g],0,bias=1),window=2)
        ax2.fill_between(II,avg-std,avg+std,color=colors[2*g+1],alpha=0.2,lw=0)
        ax2.plot(II,avg,color=colors[2*g+1],lw=2,ls='--')
    plt.ylabel('plant compactness')
    plt.ylim(ymin=0,ymax=1)
    P4D_help.xlabel_full(II,darks,midnights,days,xlabel='time'+udas,alpha=0.2)    
    plt.xlim([0,900])
    plt.savefig(project+'/fig4f_plantacc.svg') 
    #plt.show()      
    
    ############################################################### paper 5a: leaf full areas

    leafo=[3,5,7]                                                                                               # plot features of leaves 2, 4, and 6 (the first to indices are cotyledons)
    L=len(leafo)
    q=qare
    plt.clf()
    for g in range(G):
        for b in [1]+leafo:                                                                                     # here, include one cotyledon in plotting
            pa=[luarea[g][b][f] for f in range(Fg[g])]
            avg=P4D_help.sliding_median(sp.stats.nanmedian(pa,0),window=20)                                     # use median 3D areas to ignore outliers
            std=P4D_help.sliding_median(sp.stats.nanstd(pa,0,bias=1),window=20)
            plt.fill_between(II,avg-std,avg+std,color=cobs[b],alpha=0.2,lw=0)
            plt.plot(II,avg,color=cobs[b],lw=2,ls=styles[g])
    P4D_help.xlabel_full(II,darks,midnights,days,xlabel='time'+udas,alpha=0.2)    
    plt.ylabel('leaf area'+ummm)
    plt.xlim([0,900])
    plt.ylim([0,80])
    plt.savefig(project+'/fig5a_leafareas.svg')
    #plt.show() 
    
    ###################################################### paper 5b: leaf fold rer
    
    plt.clf()                                                                                                   # plot leaf RERs as 24-hour averages
    for g in GG:
        for ji,j in enumerate(leafo):
            pr=lrers[g][j]
            avg=P4D_help.sliding_median(sp.stats.nanmedian(pr,0),window=2)
            std=P4D_help.sliding_median(sp.stats.nanstd(pr,0,bias=1),window=2)
            plt.fill_between(HX,avg-std,avg+std,color=cobs[j],alpha=0.0,lw=0)
            plt.plot(HX,avg,color=cobs[j],lw=2,ls=styles[g])        
    ax=P4D_help.xlabel_fold(HH,AnaSetupTimeBin,xlabel='time'+uhou) 
    plt.ylabel('leaf RER'+upph)  
    plt.savefig(project+'/fig5b_leafrer.svg')  
    #plt.show()      

    
    ###################################################### paper 5c: leaf fold and hist
    
    plt.clf()                                                                                                   # plot leaf RERs at day and night and test for differences
    tts=[]
    for g in range(G):
        for ji,j in enumerate(leafo):
            dstack=np.hstack([ldavg[g][j][f][:7] for f in range(Fg[g])])
            nstack=np.hstack([lnavg[g][j][f][:7] for f in range(Fg[g])])
            dstack=dstack[dstack==dstack]
            nstack=nstack[nstack==nstack]
            davg=sp.stats.nanmean(dstack)
            dstd=sp.stats.nanstd(dstack,bias=1)
            navg=sp.stats.nanmean(nstack)
            nstd=sp.stats.nanstd(nstack,bias=1)
            plt.errorbar(2*g+1+2*G*ji,davg,yerr=dstd,ecolor=cobs[j],elinewidth=4,capsize=10,capthick=4,ls=styles[g],zorder=1)        
            plt.bar(2*g+1+2*G*ji,davg+1,width=0.9,lw=4,align='center',ec=cobs[j],color='white',ls=stylo[g],zorder=2,bottom=-1)    
            plt.errorbar(2*g+2+2*G*ji,navg,yerr=nstd,ecolor=cobs[j],elinewidth=4,capsize=10,capthick=4,ls=styles[g],zorder=1)        
            plt.bar(2*g+2+2*G*ji,navg+1,width=0.9,lw=4,align='center',ec=cobs[j],color='white',ls=stylo[g],zorder=2,bottom=-1)  
            if(len(dstack)==0 or len(nstack)==0):
                tt=np.nan
            else:
                tt=sp.stats.ttest_ind(dstack,nstack)[1]
            tts.append(tt)
    ymax=0.08
    for g in range(G):
        for ji,j in enumerate(leafo):
            plt.text(2*g+1.5+2*G*ji,ymax*0.9,'p='+'%.1e'%tts[g*L+ji],ha='center')      
    plt.ylim(ymin=0,ymax=ymax)
    plt.xlim([0,1+2*G*L])    
    plt.xticks(2.5+2.0*G*np.arange(3),np.subtract(leafo,1))#plt.xticks(range(1,1+2*G*3),dn*G*3)
    plt.ylabel('leaf RER'+upph)
    plt.xlabel('leaf number')
    plt.savefig(project+'/fig5c_leafhist.svg')
    #plt.show()      
    
    plt.close('all')    
    
############################################################### paper 5d: leaf full reli

    q=qzbt                                                                                                      # plot leaf base-tip z-angles as a measure for the reliability of area measurements
    plt.clf()
    for g in range(G):
        for ji,j in enumerate(leafo):
            avg=P4D_help.sliding_median(sp.stats.nanmean(luzbt[g][j],0),window=20)
            std=P4D_help.sliding_median(sp.stats.nanstd(luzbt[g][j],0,bias=1),window=20)
            plt.fill_between(II,avg-std,avg+std,color=cobs[j],alpha=0.2,lw=0)
            plt.plot(II,avg,color=cobs[j],lw=2,ls=styles[g])
    P4D_help.xlabel_full(II,darks,midnights,days,xlabel='time'+udas,alpha=0.2)    
    plt.ylabel('leaf z-angle base-tip'+udeg)
    plt.xlim([0,I])
    plt.savefig(project+'/fig5d_leafullozbt.svg')
    #plt.show()  
    
    ################################################################ paper 5e: leaf fold reli          
    
    plt.clf()                                                                                                   # plot reliability angle as 24-hour average
    for g in GG:
        for ji,j in enumerate(leafo):
            avg=P4D_help.sliding_median(sp.stats.nanmean(lozbt[g][j],0),window=2)            
            std=P4D_help.sliding_median(sp.stats.nanstd(lozbt[g][j],0,bias=1),window=2)
            plt.fill_between(HX,avg-std,avg+std,color=cobs[j],alpha=0.2,lw=0)
            plt.plot(HX,avg,color=cobs[j],lw=2,ls=styles[g])        
    ax=P4D_help.xlabel_fold(HH,AnaSetupTimeBin,xlabel='time'+uhou) 
    plt.ylabel('leaf z-angle base-tip'+udeg)      
    plt.axis(ax)
    plt.savefig(project+'/fig5e_leaffoldreli.svg')  
    #plt.show()  
    
    ################################################################ paper 5f: appearances
    
    plt.clf()                                                                                                   # plot timing of leaf appearances
    tts=[]
    for b in range(B):   
        tt=[[] for g in range(G)]
        for g in range(G):           
            prop=np.array([f for f in ltime[g][b]])#if type(f)==float
            avg=sp.stats.nanmean(prop)
            std=sp.stats.nanstd(prop,bias=1)
            plt.errorbar(G*b+g+1,avg,yerr=std,ecolor=cobs[b],elinewidth=4,capsize=10,capthick=4,zorder=1)        
            plt.bar(G*b+g+1,avg+1,width=0.9,lw=4,align='center',edgecolor=cobs[b],color='white',ls=stylo[g],zorder=2,bottom=-1) 
            tt[g]=prop[prop==prop]
        if(np.nansum(tt[0])>0 and np.nansum(tt[1])>0):
            ts=sp.stats.ttest_ind(tt[0],tt[1*(G>1)])[1]                                                             # do not perform t-test when only one genotype
        else:
            ts=0.0
        tts.append(ts)
    for b in range(B):
        plt.text(G*b+1.5,24*0.9,'p='+'%.1e'%tts[b],ha='center')      
    plt.xlabel('leaf number')
    plt.ylabel('leaf appearances'+udas)
    plt.xticks(3.5+G*np.arange(B),range(B))
    plt.xlim([4.5-0,4.5+G*(B-2)])
    plt.ylim([10,30])
    plt.savefig(project+'/fig5f_leafappearance.svg')  
    #plt.show()       
    
    ############################################################### paper 5g: leaf curv  

    plt.clf()                                                                                                   # plot leaf curvature over the full time
    for g in range(G):
        for ji,j in enumerate(leafo):
            avg=P4D_help.sliding_median(sp.stats.nanmean(lucurv[g][j],0),window=20)            
            std=P4D_help.sliding_median(sp.stats.nanstd(lucurv[g][j],0,bias=1),window=20)
            plt.fill_between(II,avg-std,avg+std,color=cobs[j],alpha=0.2,lw=0)
            plt.plot(II,avg,color=cobs[j],lw=2,ls=styles[g])    
    P4D_help.xlabel_full(II,darks,midnights,days,xlabel='time'+udas,alpha=0.2)
    plt.xlim([0,900])
    plt.ylabel('leaf curvature'+ummx)    
    plt.savefig(project+'/fig5g_leafcurv.svg')
    #plt.show() 
    
    ################################################################ paper 5h: leaf nutation          
    
    plt.clf()                                                                                                   # plot the 24-hour average leaf nutation as measured by the base-tip xy-angle
    for g in GG:
        for ji,j in enumerate(leafo):
            avg=P4D_help.sliding_median(sp.stats.nanmean(loxbt[g][j],0),window=2)  
            avg=avg-sp.stats.nanmean(avg)
            std=P4D_help.sliding_median(sp.stats.nanstd(loxbt[g][j],0,bias=1),window=2)/100.0
            plt.fill_between(HX,avg-std,avg+std,color=cobs[j],alpha=0.2,lw=0)
            plt.plot(HX,avg,color=cobs[j],lw=2,ls=styles[g])        
    ax=P4D_help.xlabel_fold(HH,AnaSetupTimeBin,xlabel='time'+uhou) 
    plt.ylabel('leaf xy-angle origin-center'+udeg)
    plt.ylim([-50,50])     
    plt.savefig(project+'/fig5h_leafnutation.svg')  
    #plt.show() 
    
    ###################################################### paper 5i: leaf amp
    
    plt.clf()                                                                                                   # plot leaf oscialltion amplitudes as functions of leaf 3D areas
    ax1=plt.subplot(111)
    ax2=ax1.twinx()
    ax1.set_ylabel('leaf z-angle base-tip'+udeg)                                                                # leaf blade angle
    ax2.set_ylabel('leaf z-angle origin-base'+udeg)                                                             # petiole angle
    colz=['black','gray']                                                                                       # colors for plotting of the two angles
    for g in range(G):
        pola=[]
        polx=[]
        poly=[]
        polz=[]
        for f in range(Fg[g]):
            for b in range(B):       
                a=luarea[g][b][f]                                                                               # 3D area
                x=lwzbt[g][b][f]                                                                                # oscillation amplitude of leaf blade angle
                y=lwzob[g][b][f]                                                                                # oscillation amplitude of petiole angle
                idx=(a==a)*(x==x)*(y==y)                                                                        # ignore NaNs
                pola.append(a[idx])
                polx.append(x[idx])
                poly.append(y[idx])
                ax1.plot(a[::10],x[::10],color=colz[0],lw=2,marker='o',ls='.',alpha=0.1,mec='none',zorder=1)    # plot every 10th point only
                ax2.plot(a[::10],y[::10],color=colz[1],lw=2,marker='s',ls='.',alpha=0.1,mec='none',zorder=1)    # plot every 10th point only     
        ha=np.hstack(pola)
        hx=np.hstack(polx)
        hy=np.hstack(poly)
        print genon[g],sp.stats.pearsonr(ha,hx)[0]**2,sp.stats.pearsonr(ha,hy)[0]**2
        polyx,statux=np.polyfit(ha,hx,1,full=1)[:2]                                                             # linear fit of leaf blade angles versus leaf 3D area
        polyy,statuy=np.polyfit(ha,hy,1,full=1)[:2]                                                             # linear fit of petiole angles versus leaf 3D area
        rr=np.arange(0,70,10)
        ax1.plot(rr,np.poly1d(polyx)(rr),color=colz[0],lw=2,ls=styles[g],zorder=10)
        ax2.plot(rr,np.poly1d(polyy)(rr),color=colz[1],lw=2,ls=styles[g],zorder=10)       
    plt.xlabel('leaf areas'+ummm)    
    plt.xlim([0,60])
    ax1.set_ylim([0,25])    
    ax2.set_ylim([0,25]) 
    plt.ylim(ymin=0)
    plt.savefig(project+'/fig5i_leafamp.svg')
    #plt.show()     
               
    ################################################################ end
    
    S=1                                                                                                                             # compute number of movies to process
    s=0                                                                                                                             # compute number of current movie
    dt=time.time()-tempo                                                                                                            # compute duration of current analysis step
    print 'multi:','time spent',dt,'seconds','time remaining',(S-s)*dt/(3600.0*MainCores),'hours'                                   # print computation time statistics
    
    return 0
        

    

