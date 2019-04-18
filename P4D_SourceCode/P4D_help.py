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
import cvxopt
import cvxopt.solvers
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy as sp
import scipy.misc
import scipy.ndimage
import scipy.optimize
import scipy.spatial
import scipy.stats
import skimage
import skimage.filters
import skimage.morphology
import skimage.feature
import skimage.measure
import sys
import time
import traceback

#import pyhull
#import pyhull.convex_hull

import P4D_help                                 # implementations of various auxiliary functions
reload(P4D_help)

###################################### convex hull

#def convex_hull_image(image):
#    '''Compute the convex hull of a binary 2d array and return as a binary 2D mask.'''
#    image = image.astype(bool)
#    coords = skimage.morphology.convex_hull.possible_hull(image.astype(np.uint8))
#    N = len(coords)
#    coords_corners = np.empty((N * 4, 2))
#    for i, (x_offset, y_offset) in enumerate(zip((0, 0, -0.5, 0.5),(-0.5, 0.5, 0, 0))):
#        coords_corners[i * N:(i + 1) * N] = coords + [x_offset, y_offset]
#    coords = skimage.morphology.convex_hull.unique_rows(coords_corners)
#    offset = coords.mean(axis=0)
#    coords -= offset
#    chull=np.array(pyhull.convex_hull.ConvexHull(coords).vertices)
#    v = coords[np.unique(chull)]
#    v_centred = v - v.mean(axis=0)
#    angles = np.arctan2(v_centred[:, 0], v_centred[:, 1])
#    v = v[np.argsort(angles)]
#    v += offset
#    mask = skimage.morphology.convex_hull.grid_points_inside_poly(image.shape[:2], v)
#    return mask

########################################## generate subfolders

def generate_subfolders(path,exceptions):    
    '''For a given path, collect all condition and experiment folders and return those that do not contain an exceptional string.'''
    folders=[]        
    conditions=np.sort(os.listdir(path))
    conditions=[c for c in conditions if not np.sum([x in c for x in exceptions])]
    for ci,condition in enumerate(conditions):
        experiments=np.sort(os.listdir(path+'/'+condition))
        experiments=[e for e in experiments if not np.sum([x in e for x in exceptions])]    
        for ei,experiment in enumerate(experiments):            
            folders.append(path+'/'+condition+'/'+experiment)   
    folders=[f for f in folders if not np.sum([x in f for x in exceptions])]
    return folders
    
############################## do jobs

def dojobs(jobs,name):
    '''Process queued parallel jobs and print computation time statistics.'''
    J=len(jobs)
    temp=time.time()    
    for index,job in enumerate(jobs):        
        job.get()
        dt=(time.time()-temp)/(index+1.0)/60.0
        print name+':',dt,'min per job',(J-index)*dt/60.0,'hours left' 
    return 0
    
############################### multiprocessing with traceback

def doexception(inq):
    '''Handle exceptions when using parallelizations with the multiprocessing module.'''
    functor,inp=inq
    try:
        return functor(inp)
    except:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))  
    
################################### filter osci amp

def fosci(y,dt=60): 
    '''For a given time series, return for each point the maximal amplitude difference within a specified window.'''       
    I=len(y)
    dy=y.copy()*0.0
    for i in range(I):
        yy=y[max(i-dt,0):min(i+dt,I)]            
        dy[i]=np.nanmax(yy)-np.nanmin(yy)            
    return dy    
    
############################## get overlap of two intervals

def getOverlap(a,b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))   
    
################################ finite difference

def deriv(avg,width):
    '''Compute finite difference of time series.'''
    X=len(avg)
    avgs=np.hstack([avg,avg,avg])
    avgd=[]
    for x in range(1*X,2*X):
        avgd.append((avgs[x+width]-avgs[x-width])/(2.0*width))
    return avgd    

############################## gaussian time series smoothing from http://wiki.scipy.org/Cookbook/SignalSmooth

def sliding_window(x,window_len=11,window='hanning'):
    '''Smooth a time series with a specified window of specified window width.'''
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:  
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='same')
    return y[window_len:-window_len+1]
    
############################## median time series smoothing

def sliding_median(x,window=3):
    '''Smooth a time series with a median filter of specified window width.'''
    res=[]
    X=len(x)
    for i in range(window,X-window):
        dat=x[i-window:i+window]
        res.append(np.median(dat[dat==dat]))
    return np.hstack([x[0:window],res,x[X-window:X]])

############################################################ restrict set of values to an interval

def bounds(x,xmi,xma):
    '''Restrict set of values to an interval.'''
    if(x<xmi):
        x=xmi
    elif(x>xma):
        x=xma        
    return x

############################################################ fit time series

def fourier(signal,dt):
    '''Return the Fourier spectrum of a time series signal with sampling interval dt.'''
    T=len(signal)
    FFT=np.array(abs(scipy.fft(signal))[0:T/2])
    freqs=np.array(scipy.fftpack.fftfreq(T,dt)[0:T/2])
    dats=sp.log10(FFT+1)
    return [freqs,dats]

def oscillation(x,freq,amp,phase,slope,offset,fall,rise):
    '''For a time vector x return an oscillation with frequency, phase whose amplitude growth with a time scale rise and then decays with a time scale fall. Furthermore, there is a linear trend with slope and offset.'''
    A=(1.0+np.exp(-x/fall)-np.exp(-x/rise))
    return A*amp*np.sin(2.0*np.pi*x*freq+phase)+slope*x+offset

def fit_oscillation(x,y):
    '''Providing x and y values, fit an oscillation to get its frequency, amplited, phase etc..'''
    fx,fy=fourier(y,x[1]-x[0])
    freq=fx[np.argmax(fy[10:])+10]
    popt,pcov=sp.optimize.curve_fit(oscillation,x,y,p0=[freq,1.0,1.0,1.0,1.0,2.0,2.0])
    plt.plot(x,y)
    plt.plot(x,oscillation(x,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6]))
    plt.plot(x,(1.0+np.exp(-x/popt[5])-np.exp(-x/popt[6])))
    plt.show()
    return popt
    
def fit_linear(x,p1,p0):
    '''Fit linear function with positive slope.'''
    return p1*(p1>0)*x+p0       
    
def fit_parabola(x,p2,p1,p0):
    '''Fit convex parabola.'''
    return p2*(p2>0)*x*x+p1*x+p0        

############################################################# compute angles and distances
    
def angle_xy(dxy):
    '''Compute xy-angle between 0 and 360 degrees.'''
    dy,dx=dxy
    rad2deg=180.0/np.pi
    return np.mod(np.arctan2(dx,dy)*rad2deg+180.0,360.0)
    
def angle_z(origin,base,dxy):
    '''Compute z-angle between two points with spacing dxy in xy-direction.'''
    return np.arctan((base[2]-origin[2])/(dxy*np.sqrt(np.sum((base[0:2]-origin[0:2])**2))))*180.0/np.pi
    
def angle_vector(x,y):
    '''Compute angle between two vectors.'''
    return np.arccos(np.dot(x,y)/np.sqrt(np.dot(x,x)*np.dot(y,y)))*180.0/np.pi    
                
def distance(base,tip,dxy):
    '''Compute distance between two points with spacing dxy in xy-direction.'''
    return np.sqrt(dxy**2*((base[0]-tip[0])**2+(base[1]-tip[1])**2)+(base[2]-tip[2])**2)
    
def angle_nojump(aa):
    '''Correct angle time series to avoid 360 degree jumps.'''
    idx=(aa==aa)
    an=aa[idx].copy()
    ad=np.arange(-3.0,3.1,1.0)*360.0     
    for i in range(1,len(an)):        
        ai=np.argmin(np.abs(an[i]-an[i-1]+ad))    
        an[i:]=an[i:]+ad[ai]       
    af=aa.copy()
    af[idx]=an
    return af
    
######################################################### colors

def hex2rgb(value):
    '''Convert hex-color to rgb-color.'''
    value = value.lstrip('#')
    lv = len(value)
    return np.array([int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)]+[255.0])/255.0

######################################################### vector distances

def vec_ortho(v,u):
    '''Return component of v which is orthogonal to u.'''
    unorm=u/sp.linalg.norm(u)
    return v-np.dot(v,unorm)*unorm
    
def vec_scale(v,dxy):
    '''Return rescaled vector with all components given in millimeters.'''
    u=v.copy()
    u[:,:2]*=dxy
    return u    
        
def pnt2plane(v0,v1,v2): 
    '''Compute minimal distance of point v0 to plane spanned by v1 and v2.'''
    vx=v0/sp.linalg.norm(v0)
    v1=v1/sp.linalg.norm(v1)
    v2=v2/sp.linalg.norm(v2)
    vo=vec_ortho(v1,v2)*10.0
    vd=np.cross(vo,v2)
    vd=vd/sp.linalg.norm(vd)        
    return vo,vd,vx.dot(vd)   
        
def pnt2line(pnt,start,end,segment=False):  
    '''Compute minimal distance of point pnt to  line (segment) from start to end.'''
    line_vec=end-start
    pnt_vec=pnt-start    
    line_len=np.sqrt(np.sum(line_vec**2))
    line_unitvec=line_vec/line_len    
    pnt_vec_scaled=pnt_vec/line_len    
    t=np.dot(pnt_vec_scaled,line_unitvec) 
    cc=np.cross(line_vec,pnt_vec)
    if(segment==True):
        t=np.where(t<0.0,0.0,t)
        t=np.where(t>1.0,1.0,t)
    nearest=np.outer(t,line_vec)
    dist=np.sqrt(np.sum((nearest-pnt_vec)**2,axis=1))*sp.sign(cc)  
    nearpoint=nearest+start        
    return dist,nearpoint
    
def distance_to_line(imMASK,s0,s1): 
    '''Return map of distances to line.'''
    wh=np.array(np.where(imMASK)).T
    dist,near=P4D_help.pnt2line(wh,s0,s1,segment=False)    
    imDIST=np.zeros(imMASK.shape)
    imDIST[np.where(imMASK)]=dist
    return imDIST
    
def distance_to_lines(c,points):  
    '''Return minimal distance of point to multiple lines.'''
    e=np.zeros(2)
    L=len(points)/2
    for l in range(L):
        a=points[2*l+0]
        d=points[2*l+1]-points[2*l+0]
        de=c-a-d*np.dot(c-a,d)/np.dot(d,d)
        e+=np.sum(de**2)
    return e    
    
def minimize_distance_to_lines(co,points): 
    '''Return point with minimal distance to multiple lines.''' 
    return sp.optimize.leastsq(P4D_help.distance_to_lines,co,args=(points))[0]      
     
############################################################# spatial profiles for detection of leaf positions    

def radial_profile(lx,ly,x0,y0):
    '''Make a 2D radial profile of size [lx,ly] centered around [x0,y0].'''
    x=np.arange(0,lx,1.0)
    y=np.arange(0,ly,1.0)
    y=y[:,np.newaxis]
    return (x-x0)**2 + (y-y0)**2

def spiral_profile(lx,ly,x0,y0):
    '''Make a 2D spiral profile (0 to 360, clock-like) of size [lx,ly] centered around [x0,y0].'''
    x=np.arange(0,lx,1.0)
    y=np.arange(0,ly,1.0)
    y=y[:,np.newaxis]
    return angle_xy([x0-x,y0-y])    

################################################################ fit leaf surface

def polyfit2d(x,y,z,x0,y0,order=3):
    '''Fit 2D landscape with polynomial with origin [x0,y0].'''
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = (x-x0)**i * (y-y0)**j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m

def polyval2d(x,y,x0,y0,m):
    '''Return 2D landscape given by a polynomial m with origin [x0,y0].'''
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = 1.0*np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += 1.0 * a * (x-x0)**i * (y-y0)**j
    return z

def fit_leaf_surface(imDEPTH,imMASK,x0,y0,order=3):
    '''Fit a given leaf surface with a polynomial.'''
    ly,lx=imMASK.shape    
    xx,yy=np.meshgrid(np.arange(lx),np.arange(ly))
    x,y,z=xx[imMASK>0],yy[imMASK>0],imDEPTH[imMASK>0]
    m=polyfit2d(x,y,z,x0,y0,order)
    wh=np.where(imMASK)
    imFIT=1.0*imMASK.copy()    
    imFIT[imMASK]=polyval2d(wh[1],wh[0],x0,y0,m)    
    return imFIT

############################################################# restricted smoothing

def restricted_gaussian_smoothing(depth,mask,sigma):
    '''A depth image is filtered with a Gaussian of width sigma within a certain region given by mask.'''
    Fdepth=1.0*scipy.ndimage.gaussian_filter(1.0*depth*mask,sigma)
    Fmask=1.0*scipy.ndimage.gaussian_filter(1.0*mask,sigma)
    return np.where(mask>0,Fdepth/Fmask,depth)

def chambolle_denoise(depth,mask):
    '''A depth image denoised within a certain region given by mask.'''
    toDenoise=np.where(mask,depth,np.nan)
    return skimage.filters.denoise_tv_chambolle(toDenoise)
    
def restricted_median_smoothing(depth,mask,sigma):
    '''A depth image is filtered with a median filter of width sigma within a certain region given by mask.'''
    if(sigma>2):
        return skimage.filters.median_filter(depth,mask=mask,radius=sigma)
    else:
        return depth        

############################################################# find first peak in time series

def peak(H,ff,unit,color):        
    mmm=scipy.signal.argrelextrema(H,np.greater)[0][:2]
    for mm in mmm:
        plt.plot([mm],[H[mm]],color=color,marker='o',ls='.')
        plt.text(mm,H[mm],'%.2f'%(mm*ff)+unit,va='bottom',ha='left')

############################################################# filter outliers

def filter_outliers(im,width=2.0):
    '''All points of an image are kept only if they are above mean + width*sd of the surrounding pixels.'''
    md=np.median(im)#np.mean(im)#
    sd=np.median(np.abs(im-md))#np.std(im)#
    L=len(im)
    if((im[L/2]-md)>width*sd):
        return 1
    else:
        return 0
    
############################################################### gray scale to virtual depth

def gray2virtual(gray):
    '''Take grayscale depth image as input and return image of virtual distances.'''
    r=2.0**16-1.0
    return 1.0/(1.0-gray/r)

############################################################### gray scale to metric depth and computing of pixelsize

def virtual2metric(virtual):
    '''Take virtual depth image as input and return image of metric distances and pixelsizes.'''
    fL=163.0 					# focal length (mm)
    TL=809.0 					# total focus distance (mm)
    B=2.75 					# distance of microlens array to sensor (mm) 		# R11: 1.6
    dx_pixel=6576.0				# sensor length 					# R11: 4016.0
    dy_pixel=4384.0 				# sensor width 						# R11: 2688.0
    size_pixel=0.0055 				# pixel size (mm) 					# R11: 0.009
    virtual_size_pixel=2.0*size_pixel 		# virtual pixel size (mm) 
    BL=TL/2.0*(1.0-np.sqrt(1.0-4.0*fL/TL))
    bL=BL+(virtual-2.0)*B
    aL=1.0/(1.0/fL-1.0/bL)
    metric=BL+aL 
    magnification=bL/aL
    pixelsize=virtual_size_pixel/magnification
    return metric,pixelsize
        
############################################################# surface area

def surface_area(depth,mask,pixelsizes):
    '''For a depth image with effective pixel sizes, compute the surface of the region given by mask.'''
    msk=mask[:-1,:-1]*mask[1:,1:]                 # both (x,y) and (x+1,y+1) must be in mask for the area to be added 
    mps=pixelsizes[:-1,:-1]
    dd=depth[1:,1:]-depth[:-1,:-1]
    dx=depth[1:,:-1]-depth[:-1,:-1]
    dy=depth[:-1,1:]-depth[:-1,:-1]
    return 0.5*np.sum(mps*msk*(np.sqrt(mps*mps+dx**2+(dx-dd)**2)+np.sqrt(mps*mps+dy**2+(dy-dd)**2)))    

############################################################# leaf orientation

def leaf_orientation(depth,mask,radial,radius):
    '''Given a depth image and a mask that represents the region of interest, compute the heights of the base-, tip-, right-, left-, center-points in mask. The heights are averaged over a disk of radius and restricted to the mask.'''  
    ly,lx=depth.shape
    leaf_xyz=[[] for i in range(5)]        
    leaf_xyz[0]=list(sp.ndimage.maximum_position(-radial,labels=mask))                      # base    
    leaf_xyz[1]=list(sp.ndimage.maximum_position(radial,labels=mask))                       # tip
    grad=P4D_help.distance_to_line(mask,np.array(leaf_xyz[0]),np.array(leaf_xyz[1]))           
    leaf_xyz[2]=list(sp.ndimage.maximum_position(grad,labels=mask))                         # right
    leaf_xyz[3]=list(sp.ndimage.maximum_position(-grad,labels=mask))                        # left
    leaf_xyz[4]=list(sp.ndimage.center_of_mass(mask))                                       # center
    for i in range(5):                                                                      # z-heights
        ball=np.zeros((ly,lx))
        a=skimage.draw.ellipse(leaf_xyz[i][0],leaf_xyz[i][1],radius,radius)
        idx=(a[0]>=0)*(a[0]<lx)*(a[1]>=0)*(a[1]<ly)            
        ball[(a[0][idx],a[1][idx])]=1
        height=np.median(depth[ball*mask>0])
        leaf_xyz[i].append(height)
    return np.array(leaf_xyz)
           
################################################ plot day-night cycle

def xlabel_full(II,darks,midnights,days,xlabel='time [day]',alpha=0.2):
        I=len(II)
        mn=np.where(midnights)[0]
        ax=plt.axis()
        plt.axis([0,I-1,ax[2],ax[3]])
        plt.fill_between(II,ax[2],ax[3],where=darks,color='black',alpha=alpha,lw=0)    
        plt.xticks(mn,(days[0]+1+np.arange(len(mn))).astype('int'))  
        plt.xlabel(xlabel)
        
def xlabel_fold(HH,TimeBin,xlabel='time [hour]',alpha=0.2):
        H=len(HH)
        ax=plt.axis()
        plt.axis([0,H-1,ax[2],ax[3]])
        plt.fill_between(range(H/2,H),ax[2]+1000,ax[3]-1000,alpha=alpha,color='black',lw=0)   
        plt.xticks((np.arange(24.0)*60.0/TimeBin).astype('int')[::4],np.divide(HH[::6],60.0/TimeBin).astype('int')[::4])
        plt.xlabel(xlabel)    
        return 0,H-1,ax[2],ax[3]

def quant_hist(sizes,I,dh,TimeBin,darks,contime):        
    di=np.argmax((contime-contime[0])>dh*60.0)
    ID=range(di,I)
    dsizes=[60.0*np.log(1.0*sizes[i]/sizes[i-di])/(contime[i]-contime[i-di]) for i in ID] 
    do=np.abs(np.diff(darks))    
    no=int(darks[0])    
    nsizes=sizes[do==1]      
    msizes=[np.log(1.0*nsizes[i]/nsizes[i-1])/12.0 for i in range(1,len(nsizes))]    
    ms_davg=msizes[(1-no)::2]
    ms_navg=msizes[no::2]
    return range(I-di),dsizes,ms_davg,ms_navg

def quant_fold(II,bins,dsizes,HH):
    fsizes=[[] for h in HH]
    H=len(HH)
    I=len(II)
    for i in range(I):
        fsizes[bins[i]].append(dsizes[i])    
    fsizes_avg=np.array([sp.stats.nanmean(h) for h in fsizes])[HH]
    fsizes_std=np.array([sp.stats.nanstd(h,bias=1) for h in fsizes])[HH]    
    fsizes_davg=np.hstack([fsizes_avg[h] for h in HH[:H/2]])
    fsizes_navg=np.hstack([fsizes_avg[h] for h in HH[H/2:]])
    return fsizes_avg,fsizes_std,fsizes_davg,fsizes_navg
    
############################################# fit plane 

def fit_plane(points): # provides center point and normal vector of 3D point cloud, see http://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points
    points=np.reshape(points,(np.shape(points)[0],-1))    
    ctr=points.mean(axis=1)
    x=points-ctr[:,None]
    M=np.dot(x, x.T)
    return ctr,np.linalg.svd(M)[0][:,-1]

def fit_plane_angle(mask,metric,pixelsizes): # provides z-angle, center point, normal vector, and points of 3D point cloud
    # mask: boolean array for region of interest
    # metric: array of metric depths
    # pixelsizes: array of (metric) pixelsizes
    dxy=np.median(pixelsizes[mask])
    wh=np.where(mask)
    x=wh[1]*dxy
    y=wh[0]*dxy
    z=metric[wh]
    points=np.vstack([x,y,z])
    ctr,vec=fit_plane(points)
    ang=np.rad2deg(np.arccos(np.dot(vec,[0,0,1])))
    return ang,ctr,vec,points
