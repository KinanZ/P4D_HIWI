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
import numpy as np
import png
import scipy as sp
import scipy.ndimage
import skimage
import skimage.draw
import skimage.filters
import skimage.filters.rank
import skimage.morphology
import skimage.feature
import skimage.measure
import skimage.segmentation
import skimage.io
import time
import cv2

import P4D_help                 # implementations of various auxiliary functions
reload(P4D_help)

def split(image):

    height, width = image.shape[:2]

    # Let's get the starting pixel coordiantes (top left)
    start_row, start_col = int(0), int(0)
    # Let's get the ending pixel coordinates (top left)
    end_row, end_col = int(height * .5), int(width * .5)
    cropped_top_left = image[start_row:end_row , start_col:end_col]

    # Let's get the starting pixel coordiantes (bottom left)
    start_row, start_col = int(height * .5), int(0)
    # Let's get the ending pixel coordinates (bottom left)
    end_row, end_col = int(height), int(width * .5)
    cropped_bot_left = image[start_row:end_row , start_col:end_col]

    # Let's get the starting pixel coordiantes (top right)
    start_row, start_col = int(0), int(width * .5)
    # Let's get the ending pixel coordinates (top right)
    end_row, end_col = int(height * .5), int(width)
    cropped_top_right = image[start_row:end_row , start_col:end_col]

    # Let's get the starting pixel coordiantes (bottom right)
    start_row, start_col = int(height * .5), int(width * .5)
    # Let's get the ending pixel coordinates (bottom right)
    end_row, end_col = int(height), int(width)
    cropped_bot_right = image[start_row:end_row , start_col:end_col]

    # remove the annoying part in the image
    height, width = cropped_top_left.shape[:2]
    start_row, start_col = int(0), int(0)
    end_row, end_col = int(height), int(width * .32)
    cropped_top_left[start_row:end_row , start_col:end_col] = cropped_top_left.mean()

    return cropped_top_left


######################################################### segment

def segment(inp):  
    '''Function segments plant and individual leaves and computes plant and leaf positions and features. Takes focus and depth images as inputs.'''
    
    f,F,i,I,depth,focus,dirD,dirF,dirM,dirN,dirP,dirS,dirV,MainPath,MainCores,SegBGupper,SegFGlower,SegSigmaGauss,SegSigmaCanny,SegThresSlope,SegThresAbsci,SegRadiusOriginHeight,SegRadiusLeafHeight,SegRadiusStemEraser,DoTrack,temp=inp    # input parameters
    tempo=time.time()                                                                                                                                                                                                                          # remember starting time
            
    for i in [i]:                                                                               # for the current image do
    
        imD = skimage.io.imread(dirD+depth)                                                     # read depth image
        #imD = 1.0*sp.ndimage.imread(dirD+depth)
        imF = 1.0*sp.ndimage.imread(dirF+focus)                                                 # read focus image

        imF = cv2.resize(imF, (imD.shape[1],imD.shape[0]))

        imD = split(imD)
        imF = split(imF)

        #print "imD", type(imD), imD.shape
        #print "imF", type(imF), imF.shape

        #pli.imsave("debug/depth"+str(i).zfill(6)+'.png', 1.0*imD)
        #pli.imsave("debug/focus"+str(i).zfill(6)+'.png', 1.0*imF)

        ################################################### segmentation plant

        ly,lx=np.shape(imF)                                                                     # get image shape
        imS=skimage.filters.edges.sobel(imF)                                                    # find weighted edge image using Sobel filter
        mark=np.zeros_like(imS)                                                                 # create auxilliary array for watershed seed points
        mark[imF<SegBGupper]=1                                                                  # set low intensity pixels as background
        mark[imF>SegFGlower]=2                                                                  # set high intensity pixels as foreground

        #pli.imsave("debug/sobel_on_focus"+str(i).zfill(6)+'.png', 1.0*imS)                                                                             
        
        imWS=(skimage.morphology.watershed(imS,mark)-1)>0                                       # perform watershed segmentation using seed points and Sobel-filtered image  
        imL,N=sp.ndimage.label(imWS)                                                            # label connected components
        coms=np.array(sp.ndimage.center_of_mass(imWS,imL,range(1,1+N)))                         # compute center of mass for each labeled component
        dists=np.sqrt(np.sum(np.subtract([0.5*ly,0.5*lx],coms)**2,1))                           # compute distances of components to middle of the image

        #pli.imsave("debug/watershed_bool_bigger_than_0_on_sobel"+str(i).zfill(6)+'.png', 1.0*imWS)
        
        mask=dists>600.0                                                                        # chose distances that exceed 600 pixels
        imCL=imWS.copy()                                                                        # create auxilliary array
        imCL[mask[imL-1]]=0                                                                     # remove small far-off components
        imL,N=sp.ndimage.label(imCL)                                                            # label remaining components    
        sizes=sp.ndimage.sum(imCL,imL,range(1,N+1))                                             # compute pixelnumber of each component      
        imPL=skimage.morphology.remove_small_objects(imCL,np.sort(sizes)[-1])                   # remove all but the largest component
        
        #pli.imsave("debug/remove_small_objects_on_water_shed"+str(i).zfill(6)+'.png', 1.0*imPL)
        
        wh=np.where(imPL)                                                                       # get coordinates of a plant pixels
        b0,b1,a0,a1=min(wh[0]),max(wh[0]),min(wh[1]),max(wh[1])                                 # compute bounding box of plant
        bd,ad=b1-b0,a1-a0                                                                       # compute edge lengths of bounding box
        
        plt.clf()                                                                              # plotting functions for testing purposes
        plt.imshow(1.0*imF,interpolation='nearest',alpha=1.0,origin='lower',cmap='gray')    
        plt.imshow(np.where(imPL,1.0,np.nan),interpolation='nearest',alpha=0.5,origin='lower')
        #plt.show()  
        plt.savefig("debug/first_original_check"+str(i).zfill(6)+'.png')

        ################################################### find center of mass and its height
                
        imSM=P4D_help.restricted_gaussian_smoothing(imD,~imPL,SegSigmaGauss)                            # smooth depth image with a Gaussian filter

        #pli.imsave("debug/gaussian_smoothing_on_depth+removed_small_objects"+str(i).zfill(6)+'.png', 1.0*imSM)
        
        ground,pixelsizes=P4D_help.virtual2metric(P4D_help.gray2virtual(imSM))                          # compute depth image to metric distances         
        middle=list(sp.ndimage.center_of_mass(imPL))                                                    # compute x-y-middle of plant via center of mass 
        ball=np.zeros((ly,lx))                                                                          # create auxilliary array to compute height of middle                     
        idxs=skimage.draw.ellipse(middle[0],middle[1],SegRadiusOriginHeight,SegRadiusOriginHeight)      # create ellipse around middle
        idx=(idxs[0]>=0)*(idxs[0]<ly)*(idxs[1]>=0)*(idxs[1]<lx)                                         # restrict to image                 
        ball[(idxs[0][idx],idxs[1][idx])]=1                                                             # draw ellipse around middle of the plant                                                
        mask=(ball*(~imPL))>0                                                                           # restrict ellipse to non-plant pixels
        middle.append(np.median(ground[mask]))                                                          # compute z-height of middle
        middle=np.array(middle)                                                                         # convert middle coordinates to array
        imRA=np.sqrt(P4D_help.radial_profile(lx,ly,middle[1],middle[0]))                                # generate radial profile around middle
        
        #pli.imsave("debug/draw_ellipse_on_removed_small_objects"+str(i).zfill(6)+'.png', 1.0*imRA)
        
        plt.clf()                                                                                     # plotting functions for testing purposes
        plt.imshow(imPL,interpolation='nearest',alpha=0.5,origin='lower')
        plt.imshow(mask,interpolation='nearest',alpha=0.5,origin='lower')
        plt.imshow(imRA,interpolation='nearest',alpha=0.5,origin='lower')
        plt.plot(middle[1],middle[0],marker='s',ls='dotted',color='white')
        #plt.axis([x0,x1,y0,y1])
        #plt.show()
        plt.savefig("debug/second_original_check"+str(i).zfill(6)+'.png')
        
        ################################################### segment leafs
                
        imPB=skimage.segmentation.find_boundaries(imPL)                                                         # get boundary of plant

        #pli.imsave("debug/find_boundaries_removed_small_objects"+str(i).zfill(6)+'.png', 1.0*imPB)
        
        imPR=skimage.morphology.remove_small_objects(imPB,connectivity=2)                                   # remove small connected components
        
        #pli.imsave("debug/imPR"+str(i).zfill(6)+'.png', 1.0*imPR)
        
        ones=np.ones((3,3))                                                                                     # create auxilliary array
        disk=skimage.morphology.disk(SegRadiusStemEraser)                                                       # create auxilliary array 
        imE=skimage.feature.canny(imF.astype('uint16'),sigma=SegSigmaCanny)                                                      # compute binary edge image using Canny filter    

        #pli.imsave("debug/canny_on_focus"+str(i).zfill(6)+'.png', 1.0*imE)
        
        imA=(imE*imPL+imPR)>0                                                                                   # combine plant boundaries and inner edges    

        #pli.imsave("debug/canny_x_removed_small_object_+_boundaries_bool_positive"+str(i).zfill(6)+'.png', 1.0*imA)
        
        imEDT=sp.ndimage.distance_transform_edt(~imA)*imPL                                                      # compute distance transform of combined edge image
        
        #pli.imsave("debug/distance_transform_of_combined_edge_image"+str(i).zfill(6)+'.png', 1.0*imEDT)

        imX2=skimage.feature.peak_local_max(imEDT,min_distance=15,indices=0,exclude_border=0)                   # find local maxima at smaller distances
        
        #pli.imsave("debug/local_maxima_at_smaller_distances"+str(i).zfill(6)+'.png', 1.0*imX2)
        
        imX6=skimage.feature.peak_local_max(imEDT,min_distance=30,indices=0,exclude_border=0)                   # find local maxima at larger distances
        
        #pli.imsave("debug/local_maxima_at_larger_distances"+str(i).zfill(6)+'.png', 1.0*imX6)
        
        imX=np.where(imRA<100.0,imX2,imX6)                                                                      # keep closer maxima towards the middle and more distant maxima away from the middle of the plant
        
        #pli.imsave("debug/where_imRA_smaller_larger"+str(i).zfill(6)+'.png', 1.0*imX)
        
        imSD1=skimage.filters.rank.maximum(imX,disk)>0                                                          # expand maxima to provide more robust seed points for watershed segmentation
        
        #pli.imsave("debug/expand_maxima"+str(i).zfill(6)+'.png', 1.0*imSD1)
        
        imSD2=imEDT>(imRA*SegThresSlope+SegThresAbsci)                                                          # binarise distance transform with threshold that increases with distance from the middle of the plant
        
        #pli.imsave("debug/threshold_on_distance"+str(i).zfill(6)+'.png', 1.0*imSD2)

        imL,N=sp.ndimage.label((imSD1+imSD2)>0,ones)                                                            # label the combined image of local maxima and thresholded distance transform
        imY=-1.0*skimage.filters.gaussian(1.0*imEDT*imPL,1.0)*imPL*imRA                                         # smooth the distance transform to fill out the whole plant and bias it towards to middle of the plant         
        
        #pli.imsave("debug/smooth_distance_transform"+str(i).zfill(6)+'.png', 1.0*imY)
        
        imSEG=skimage.morphology.watershed(imY,imL,mask=imPL)                                                   # perform the watershed segmentation     
        
        #pli.imsave("debug/watershed_segmentation"+str(i).zfill(6)+'.png', 1.0*imSEG)
        
        L=imSEG.max()                                                                                           # compute the number of labeled leaves
        
        #print "num of leaves:", L
        
        for li,l in enumerate(range(1,L+1)):                                                                    # remove all labeled components with only few pixels
            imMASK=(imSEG==l)
            if(imMASK.sum()<4):
                imSEG[imMASK]=0
        imSEG=skimage.segmentation.relabel_sequential(imSEG)[0]                                                 # relabel leaves
        L=imSEG.max()                                                                                           # compute the number of labeled leaves
        #pli.imsave(dirM+'mats_t='+str(i).zfill(6)+'.png',1.0*imSEG,cmap='gray',origin='lower',vmin=0,vmax=255)  # export image of segmented plant and leaves                             

        plt.clf()                                                                                              # plotting functions for testing purposes
        plt.imshow(imF,origin='lower',cmap='gray',alpha=0.5) 
        plt.imshow(1.0*imL,origin='lower',alpha=0.5)
        plt.imshow(imA,origin='lower',cmap='jet',alpha=0.5) 
        plt.imshow(imSEG,origin='lower',cmap='jet',alpha=0.5)
        plt.savefig("debug/third_original_check"+str(i).zfill(6)+'.png') 
#        plt.show()    

        ######################################################### compute leaf properties
            
        quants=np.zeros((L,5))                                                                                  # array for leaf features
        coords=np.zeros((L,5,3))                                                                                # array for leaf positions
        imSM=P4D_help.restricted_gaussian_smoothing(imD,imPL,SegSigmaGauss)                                     # smooth depth image
        plant,pixelsizes=P4D_help.virtual2metric(P4D_help.gray2virtual(imSM))                                   # compute metric depth image and pixelsizes
        imSHOW=imF.copy()*0
        #print "enumerate(range(1,L+1))", list(enumerate(range(1,L+1)))                                                                             # create mask for leaf blades
        for li,l in enumerate(range(1,L+1)):                                                                    # for each leaf do     
            imMASK=(imSEG==l)                                                                                   # array for leaf mask               
            imLEAF=skimage.filters.rank.minimum(imMASK,disk)>0                                                   # erode leaf to remove petiole    
            if(imLEAF.sum()>0):                                                                                 # dilate leaf if still existent
                imLEAF=skimage.filters.rank.maximum(imLEAF,disk)>0                       
            else:                                                                                               # take uneroded leaf otherwise
                imLEAF=imMASK
            imSHOW[imLEAF]=l                                                                                    # set leaf mask
            touch=1.0*(skimage.segmentation.clear_border(imMASK.copy()).sum()==0)                               # check if leaf touches image border
            perim=1.0*skimage.measure.perimeter(imLEAF,neighbourhood=8)                                         # compute leaf perimeter
            chull=1.0#*P4D_help.convex_hull_image(imLEAF).sum()                                                 # compute pixelnumber of leaf convex hull
            size=sp.ndimage.minimum_filter(imLEAF,2).sum()                                                      # compute leaf 2D area 
            area=P4D_help.surface_area(plant,imLEAF,pixelsizes)                                                 # compute leaf 3D area
            quants[li]=[touch,perim,chull,size,area]                                                            # set leaf features
            coords[li]=P4D_help.leaf_orientation(plant,imLEAF,imRA,SegRadiusLeafHeight)                         # compute and set leaf positions
            
    ############################################################################# plant origin   
                        
        idx=np.argsort([q[-1] for q in quants])[-4:]                                                            # get four largest leaves
        points=coords[idx,:2,:2]                                                                                # get base and tip positions
        points=np.vstack(points)
        origin=list(P4D_help.minimize_distance_to_lines(middle[:2],points))                                     # compute leaf origin as point with minimal distance to the four leaf axes
        ball=np.zeros((ly,lx))                                                                                  # auxilliary array to compute height of origin                                                                       # create auxilliary array               
        idxs=skimage.draw.ellipse(origin[0],origin[1],SegRadiusOriginHeight,SegRadiusOriginHeight)              # create ellipse around middle
        idx=(idxs[0]>=0)*(idxs[0]<ly)*(idxs[1]>=0)*(idxs[1]<lx)                                                 # restrict to image                 
        ball[(idxs[0][idx],idxs[1][idx])]=1                                                                     # draw ellipse around middle of the plant           
        mask=(ball*(~imPL))>0                                                                                   # mask around origin restricted to non-plant
        origin.append(np.median(ground[mask]))                                                                  # compute metric height of origin
        origin=np.array(origin)         
    
      ############################################################################# plot positions
    
        plt.clf()                                                                                                                   # plot segmented leafs and positions
        plt.imshow(imF,origin='lower',alpha=1.0,cmap='gray',interpolation='nearest')    
        plt.imshow(np.where(imSEG==0,np.nan,imSEG),origin='lower',alpha=0.1,cmap='jet',vmin=1,vmax=L,interpolation='nearest')   
        plt.imshow(np.where(imSHOW==0,np.nan,imSHOW),origin='lower',alpha=0.4,cmap='jet',vmin=1,vmax=L,interpolation='nearest')   
        for l in range(L):
            plt.plot(coords[l][:,1],coords[l][:,0],ls='dotted',marker='o',color='white',ms=5.0)            
        plt.plot(origin[1],origin[0],ls='dotted',marker='^',color='white',ms=7.0)   
        plt.plot(middle[1],middle[0],ls='dotted',marker='s',color='white',ms=5.0)   
        plt.axis([a0-0.1*ad,a1+0.1*ad,b0-0.1*bd,b1+0.1*bd])        #[550,1250,550,1250])#
        plt.axis('off')        
        plt.tight_layout()
        plt.savefig(dirP+'pics_t='+str(i).zfill(6)+'.png')     
#        plt.show()
                                
        ############################################################################# save data
        
        dxy=np.median(pixelsizes[imPL])                                                                     # compute average plant pixelsize to emulate 2D area measurements
        total_size=sp.ndimage.minimum_filter(imPL,2).sum()                                                  # compute plant pixelnumber
        total_area=P4D_help.surface_area(plant,imPL,pixelsizes)                                             # compute plant 3D area
        total_touch=1.0*(skimage.segmentation.clear_border(imPL.copy()).sum()==0)                           # check if plant touches image border
        total_perim=1.0*skimage.measure.perimeter(imPL,neighbourhood=8)                                     # compute plant perimeter
        total_chull=1.0#*P4D_help.convex_hull_image(imPL).sum()                                             # compute plant convex hull        
        info=[lx,ly,dxy,L,total_size,total_area,total_touch,total_perim,total_chull]                        # set plant features        
        np.save(dirN+'nums_t='+str(i).zfill(6)+'.npy',[info,origin,middle,quants,coords,[0]])               # save plant and leaf data, without [0] some images yield a ValueError: setting an array element with a sequence.
        
        S=F*I                                                                                                                           # compute number of total images to process
        s=f*I+i                                                                                                                         # compute number of current image
        dt=time.time()-tempo                                                                                                            # compute duration of current segmentation step
        print 'segment:','folder',f+1,F,'image',i+1,I,'time spent',dt,'seconds','time remaining',(S-s)*dt/(3600.0*MainCores),'hours'    # print computation time statistics
            
    return 0


