import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from skimage.color import rgb2gray

from skimage.filter import gabor_kernel
from skimage import exposure
from scipy import ndimage as nd

from sklearn import decomposition
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import tree
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

import datetime
import collections
import cv2
import pickle
import operator
import os, sys, inspect

#-------------------------------------------CLASS getData-------------------------------------------

class getData(object):
    """handles all the input required, and initial data munging
    """
    
    def __init__(self, machine="laptop", announce = 0):
        """Because this project was developed on a number of different machines outside of the vagrant environment
        it became necessary to define a set of directory paths so the code would run in each environment
        """
        
        #set the machine
        self.machine = machine
        if announce == 1:
            print "You are working on", self.machine
            
        self.numberImagesPerClass = 5000
            
        #using the vagrant distribution    
        if(self.machine == "laptop"):
            self.labelPath = "/home/vagrant/Downloads/GAPROJECT/TRAIN/"
            self.filePath = "/home/vagrant/Downloads/GAPROJECT/TRAIN/train/"
            self.testFilePath = "/home/vagrant/Downloads/GAPROJECT/TEST/"
            self.resultsCPath = "/home/vagrant/Downloads/GAPROJECT/RESULTS/CASCADE/"
            self.resultsTPath = "/home/vagrant/Downloads/GAPROJECT/RESULTS/TENCLASS/"
            self.picklePath = "/home/vagrant/Downloads/GAPROJECT/PICKLE/"
            
        #running on a desktop machine
        elif(self.machine == "desktop"):
            self.labelPath = "/Users/mrgh/Desktop/GAP/TRAIN/"
            self.filePath = "/Users/mrgh/Desktop/GAP/TRAIN/train/"
            self.testFilePath = "/Users/mrgh/Desktop/GAP/TEST/"
            self.resultsCPath = "/Users/mrgh/Desktop/GAP/RESULTS/CASCADE/"
            self.resultsTPath = "/Users/mrgh/Desktop/GAP/RESULTS/TENCLASS/"
            self.picklePath = "/Users/mrgh/GA_PYTHON_2/"
            
        #using my wife's iMac!
        elif(self.machine == "wdesktop"):
            self.labelPath = "/Users/markholt/Desktop/TRAIN/"
            self.filePath = "/Users/markholt/Desktop/TRAIN/train/"
            self.testFilePath = "/Users/markholt/Desktop/TEST/"
            self.resultsPath = "/Users/markholt/Desktop/RESULTS/"
    
    def getPath(self, p):
        """return a path to the user. test is where the test set is stored, label is where the labels are stored,
        train is the directory path to the trianing set. resultsC stores the classifiers trained for the cascade models, 
        resultsT stores the classifiers for the ten class models. pickle is a general storage area.
        """
        if p == "test":
            return self.testFilePath
        elif p == "label":
            return self.labelPath
        elif p == "train":
            return self.filePath
        elif p == "resultsC":
            return self.resultsCPath
        elif p == "resultsT":
            return self.resultsTPath
        elif p == "pickle":
            return self.picklePath
    
    def readLabels(self):
        """reads the training labels from disk into memory. Returns a dataframe
        """
        myLabels = pd.read_csv(self.labelPath + "SmallObjectImages_Labels.csv")
        return myLabels
    
    def readTestLabels(self):
        """reads the test labels from disk into memory. Returns a dataframe
        """
        myLabels = pd.read_csv(self.testFilePath + "test_labels.csv")
        return myLabels
    
    def readImages(self, tNumber):
        """reads tNumber of training images into memory. Returns the list of images
        """
        imageList = []
        for i in range(1, tNumber + 1):
            #constuct the image name from the integer number
            fileName = self.filePath + str(i) + ".png"
            im1 = mpimg.imread(fileName)
            #build up the list of training images
            imageList.append(im1)   
        return imageList
    
    def readTestImages(self, tNumber):
        """reads a tNumber of test images into memory. Returns a list of images
        """
        imageList = []
        for i in range(1, tNumber + 1):
            fileName = self.testFilePath + str(i) + ".png"
            im1 = mpimg.imread(fileName)
            #build up the list of test images
            imageList.append(im1)   
        return imageList
    
    def readListOfImages(self, imList):
        """reads the images from disk into memory whose numbers are provided in imList. Returns a list of images.
        """
        imageList = []
        #work through the user provided list of images - a list of numbers
        for i in imList:
            fileName = self.filePath + str(i) + ".png"
            im1 = mpimg.imread(fileName)
            imageList.append(im1)   
        return imageList
                   
    def readOneImage(self, imageNumber):
        """Given a single image number read the training image into memory from disk and return it to the user.
        """
        fileName = self.filePath + str(imageNumber) + ".png"
        im1 = mpimg.imread(fileName)
        return im1
    
    def readOneTestImage(self, imageNumber):
        """Given a single image number read the test image into memory from disk and return it to the user.
        """
        fileName = self.testFilePath + str(imageNumber) + ".png"
        im1 = mpimg.imread(fileName)
        return im1
    
    def readImagesOfType(self, tType):
        """Given a type of image (i.e. a class label, such as "frog", or "cat") find all the training image numbers for that class.
        Return a dataframe with all the image numbers for the class, and include a column of labels (which will be all the same!)
        """
        #read in all the training labels
        myLabels = self.readLabels()
        
        #sort the labels by class
        myLabels = myLabels.sort_index(by='label').copy()
        
        #there are 5000 images of each class. Create a new dataframe, with two columns
        imagesOfTypeDf = pd.DataFrame(index=range(self.numberImagesPerClass), columns=["image_no", "label"])
        imagesOfTypeDf["label"] = tType
        
        #now fill the image_no column
        imagesOfTypeDf["image_no"] = myLabels[myLabels['label']==tType]["id"].values
        
        #return the sorted dataframe
        return imagesOfTypeDf.sort_index(by="image_no")
    
    def pickRandomSet(self, req, numClasses, classifier):
        """Routine used to set up the training sets for the cascade classifiers.
        Not a good way of doing this!!
        req is a list of class categories required - e.g. ["frog", "truck"]
        numClasses is the number of classes required - i.e. the length of req list
        classifier indicates the classifier for which the training set is being assembled for
        """
        
        #the results are stored in 2 list - the list of labels and the list of images.
        imLabList = []
        imList = []
        
        #iterate through the classes
        for klas in req:
            
            #obtain a dataframe of the images of the required class
            reqIm = self.readImagesOfType(klas)
            
            #mask is a list of image file name numbers e.g. 1.png
            mask = reqIm['image_no'].values
            ranIm = np.random.choice(mask, req[klas])
            for i in ranIm:
                im = self.readOneImage(i)
                imList.append(im)

                imLab = np.zeros(req[klas])
                
            for i in range(len(imLab)):
                if numClasses == 2:
                    if classifier == "A1":
                        imLab = self.transformLabelsForTwoClassesA1(klas)
                    else:
                        imLab = self.transformLabelsForTwoClasses(klas)
                elif numClasses == 3:
                    if classifier == "A1b":
                        imLab = self.transformLabelsForThreeClassesA1b(klas)
                    elif classifier == "A1a":
                        imLab = self.transformLabelsForThreeClassesA1a(klas)
                elif numClasses == 4:
                    if classifier == "A12":
                        imLab = self.transformLabelsForFourClassesA2(klas)
                else:
                    imLab = self.transformLabels(klas)
                imLabList.append(imLab)
                
        retList = []
        retList.append(imLabList)
        retList.append(imList)
        
        return retList
    
    
    def transformLabels(self, x):
        """This takes in the class label (the full string - such as frog), and returns the an integer equivalent to indicate class
        """
        
        if x == "frog":
            return 0
        elif x == "truck":
            return 1
        elif x == "deer":
            return 2
        elif x == "automobile":
            return 3
        elif x == "bird":
            return 4
        elif x == "cat":
            return 5
        elif x == "dog":
            return 6
        elif x == "horse":
            return 7
        elif x == "ship":
            return 8
        elif x == "airplane":
            return 9
        
    def transformLabelsForTwoClasses(self, x):
        """Produces 0 or 1 for a two class problem. 0 represents the animals, and by consequence 1 represents mechanical
        """
        
        if x == "frog" or x == "deer" or x == "bird" or x == "cat" or x == "dog" or x == "horse":
            return 0
        else:
            return 1
        
    def transformLabelsForTwoClassesA1(self, x):
        """Produces 0 or 1 for a two class problem. 0 represents frog, deer or horse, and by consequence 1 represents bird, cat or dog
        """
        
        if x == "frog" or x == "deer" or x == "horse":
            return 0
        elif x == "bird" or x == "cat" or x == "dog":
            return 1
        
    def transformLabelsForThreeClassesA1b(self, x):
        """Produces 0,1 or 2 for a three class problem. 0 represents frog, 1 deer, and 2 horse
        """
        
        if x == "frog":
            return 0
        elif x == "deer":
            return 1
        elif x == "horse":
            return 2
        
    def transformLabelsForThreeClassesA1a(self, x):
        """Produces 0,1 or 2 for a three class problem. 0 represents cat, 1 dog, and 2 bird
        """
        
        if x == "cat":
            return 0
        elif x == "dog":
            return 1
        elif x == "bird":
            return 2
        
    def transformLabelsForFourClassesA2(self, x):
        """Produces 0, 1, 2, or 3 for the 4 class mechanical problem. 0 represents truck, 1 automobile, 2 ship and 3 airplane
        """
        
        if x == "truck":
            return 0
        elif x == "automobile":
            return 1
        elif x == "ship":
            return 2
        elif x == "airplane":
            return 3
                
    def saveData(self, data, fileName, path):
        """General routine for saving results and any data structure to disk, using pickle. 
        User provides the filename and directory path category.
        The results path category is generally used to store classifiers and their parameters
        """
        
        if path == "results":
            thePath = self.resultsPath
        else:
            thePath = path
        with open(thePath + fileName + ".pik", "wb") as f:
            pickle.dump(data, f, -1)
                        
    def retrieveData(self, fileName, path):
        """General routine for retrieving results and any data stored on disk, using pickle. 
        User provides the filename and directory path category
        """
        
        if path == "results":
            thePath = self.resultsPath
        else: 
            thePath = path
        with open(thePath + fileName, "rb") as f:
            data = pickle.load(f)
        return data
    
    
#-------------------------------------------CLASS preProcess-------------------------------------------
    
    
class preProcess(object):
    """provides a set of basic pre-processing routines
    """
    
    def convertToGray(self, imageList):
        """converts a list of color images to grayscale, and returns a list of grayscale images
        """
        
        imageGrayList = []
        for image in imageList:
            imG = rgb2gray(image)
            imageGrayList.append(imG)
        return imageGrayList
    
    def whitenGrayImage(self, imageList):
        """whitens a list of grayscale images - meaning zero mean unit standard deviation. Returns a list of whitened images
        """
        
        imageWList = []
        for image in imageList:
            imageN = (image - image.mean()) / image.std()
            imageWList.append(imageN)
        return imageWList
    
    def globalHistEq(self, imageList):
        """performs histogram equalization on a list of images. Returns a list of so transformed images
        """
        
        gHEList = []
        for image in imageList:
            imageE = exposure.equalize_hist(image)
            gHEList.append(imageE)
        return gHEList
    
    def colorSplit(self, imageList, numChannels=3):
        """Splits up a color image into its constituent red, green, and blue components.
        By setting numChannels = 4 will handle an image with an alpha channel
        Any alpha channel is discarded
        Processes a list of color images, and returns an ordered dictionary of the lists of the color channels.
        """
        
        #have separate lists for each channel
        rList = []
        gList = []
        bList = []
        
        #initiate the dictioinary
        colorList = collections.OrderedDict()
        
        #iterate through the image list
        for image in imageList:
            
            #match the array split with the correct number of indices
            if numChannels == 3:
                b, g, r = cv2.split(image)
            elif numChannels == 4:
                b, g, r, a = cv2.split(image)
                
            #build the individual color channel lists    
            bList.append(b)
            gList.append(g)
            rList.append(r)
           
        #create the dictionary for return    
        colorList["red"] = rList
        colorList["green"] = gList
        colorList["blue"] = bList
        return(colorList)
    
    
    
#-------------------------------------------CLASS featureExtraction-------------------------------------------    
    
    
class featureExtraction(object):
    """undertakes other, more generic, feature extraction
    """
    
    def cannyFilter(self, imageList, tSigma = 2.5):
        """Applies the canny filter algorithm to a list of images. Returns the filtered images in a list.
        """
        imageCannyList = []
        for image in imageList:
            
            #get the canny filter from the skimage library
            imageC = filter.canny(image, sigma=tSigma)
            imageCannyList.append(imagec)
        return imageCannyList    
    
    
    
#-------------------------------------------CLASS gaborFeatureExtraction-------------------------------------------
    

class gaborFeatureExtraction(object):
    """Undertakes all feature extraction for using Gabor Filter Banks. This does the heavy lift for the project.
    """
    
    def __init__(self):
        """Set some instance variables
        """
        
        #the images are 32 by 32 pixels
        self.imageDim = 32
        
        #the number of frequency bands
        self.numberOfBands = 7
        self.numberOfBandsByTwo = 4
        
        #the number of orientation angles to use
        self.numberOfAngles = 8
        
    def power(self, image, kernel):
        """Convolution of the Gabor kernel with the image using both real and imaginary kernel parts
        """
        
        # Normalize images for better comparison.
        image = (image - image.mean()) / image.std()
        
        # Convolve the real and imaginary parts of the Gabor kernel separately. Square the result and then add the two parts.
        # Return the the square root of the sum of the squares.
        # Ensure positive real values
        return np.sqrt(nd.convolve(image, np.real(kernel), mode='wrap')**2 +
                       nd.convolve(image, np.imag(kernel), mode='wrap')**2)
    
        
    def buildGaborFilterBanks(self, useOpenCV=1):
        """Produces a set of Gabor filter kernels of varying orientations, sizes and frequencies
        """
        
        #use an oredered collection to keep the filter kernels in order
        filters = collections.OrderedDict()
        
        #the spatial aspect ratio, 0.3 is an accepted value
        gamma = 0.3
        
        #ksi represents a "band". There are 7 bands 3 & 5, 7 & 9, 11, & 13, 15 & 17, 19 & 21, 23 & 25, 27 & 29
        #ksize is the filter pixel size, 5 means a 5 by 5 pixel filter
        #Lambda is the wavelength (1/frequency) of the filter
        for (ksi, ksize, lam) in ((3, 5, 0.8), (5, 5, 1.6), 
                                 (7, 5, 2.4), (9, 5, 3.2), 
                                 (11, 7, 1.6), (13, 7, 2.4), 
                                 (15, 7, 3.2), (17, 7, 4.0), 
                                 (19, 11, 2.4), (21, 11, 3.2), 
                                 (23, 11, 4.0), (25, 11, 4.8), 
                                 (27, 15, 3.2), (29, 15, 4.0)):
            
            #Gabor operates over a Guassian "window", sigma is the standatd deviation of the Guassian
            #An accepted value is to be 0.8 * the wavelength
            sigma = 0.8 * lam
            
            i=0
            for theta_degrees in (0., 22.5, 45., 67.5, 90., 112.5, 135., 157.5):
                #convert to radians
                theta = theta_degrees * np.pi / 180.
                
                i=i+1
                indexName = "A" + str(i) + "S" + str(ksi)
                
                #there are 2 Gabor implementations availabe - one in opencv2 and one in skimage
                #use the opencv2 implementation as it allows user input to the filter size
                if(useOpenCV == 1):
                    #print "opencv kernel"
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, gamma, 0, ktype=cv2.CV_32F)
                    
                    #a recommeded scaling procedure
                    kernel /= 1.5*kernel.sum()
                else:
                    #print "skimage kernel"
                    kernel = gabor_kernel(lam, theta=theta, sigma_x=sigma, sigma_y=sigma)
                
                #accumulate the kernels in the filters list
                filters[indexName] = kernel
        return filters
        
    def process(self, img, filters, conv=0):
        """This function performs the convolution of the filter with the image. 
        There were 2 main filter (convolution) functions available - opencv's filter2 and numpy's convolve
        """
        
        #Maintaining order is useful
        fimgL = collections.OrderedDict()
        
        for idx, kernel in filters.items():
            if(conv == 1):
                #use the opencv filter routine for the convolution
                #print "opencv filter"
                fimg = cv2.filter2D(img, -1, kernel)
            else:
                #or use the numpy convolve routine
                #print "np convolve"
                fimg = self.power(img, kernel)
                
            #accumulate the filtered images    
            fimgL[idx] = fimg
        return fimgL
    
    def getC1(self, neighborhoodSize, overlap, indx, fimgL):
        """Pool the results of Gabor filtering into a small square grid, the size of which is determineed by "neighborhoodsize"
        Overlap determines the level of overlap the grid has as it moves over the image.
        """
        
        #put the results into a list
        grid = []
        
        #range over the image using the grid. The grid is overlapped
        for zx in range(0, self.imageDim - 1, overlap):
            for zy in range(0,self.imageDim - 1, overlap):
                
                #within the neighborhood the aim is to find the pixels responding the most after Gabor filtering
                max = -np.inf
                
                #Sample the pixels over the neighborhood
                for x in range(neighborhoodSize):
                    for y in range(neighborhoodSize):

                        if (x+zx)<self.imageDim and (y+zy)<self.imageDim:
                            if fimgL[indx][x+zx][y+zy] > max:
                                max = fimgL[indx][x+zx][y+zy]
                                
                #store the maximum firing pixel from the Gabor filter in the grid list
                grid.append(max);
                
        return grid
    
    def getBandGrid(self, a, b, neighborhoodSize, overlap, fimgL):
        """Produce a grid - using "getC1". Iterate over all the orientation angles, but mindful to retain which orientations
        the results come from
        """
        
        bandGrid = collections.OrderedDict()
        
        #a band consists of two sets of closely related Gabor filters
        for i in range(a, b):
            indx = fimgL.keys()[i]
            bandGrid[indx] = self.getC1(neighborhoodSize, overlap, indx, fimgL)
        
        #return the ordered dictionary of grids. Each orientation angle will have a grid
        return bandGrid   

    def maxOfGrids(self, grida, gridb):
        """Using a pair of bands, from which a pair of grids is derived, collpase the two grids into a single grid.
        """
        
        dim = len(grida)
        gridF = np.zeros(dim)
        
        #choose each pixel from the 2 grids from each band - choose the maximum value
        for x in range(dim):
            if grida[x] > gridb[x]:
                gridF[x] = grida[x]
            else:
                gridF[x] = gridb[x]
                
        #return the single grid of maximum values       
        return gridF
    
    def processC1(self, b, bandNumber):
        """Create the pooled bands
        """
        
        clayer = collections.OrderedDict()
        for i in range(self.numberOfAngles):
            idxName = b.keys()[i][0:2] + "Band" + str(bandNumber)
            #print idxName, b.keys()[i], b.keys()[i+self.numberOfAngles]
            clayer[idxName] = self.maxOfGrids(b[b.keys()[i]], b[b.keys()[i+self.numberOfAngles]])
    
        return clayer
        
    def processAllBands(self, fimgL):
        """Obtain the 14 grids and then pool them 
        """
        
        iL = []
        for i in range(self.numberOfAngles):
            nL = i * self.numberOfAngles * 2
            iL.append(nL)
        #print iL
        #print len(iL)
        
        #get the bands, the two "magic" numbers represent neighborhood size and overlap and must be tuned
        #very hard to grid search over these parameters so this was done by hand
        #each b contains the grids for the pair or orientation angles
        b1=self.getBandGrid(iL[0], iL[1], 8, 4, fimgL)
        b2=self.getBandGrid(iL[1], iL[2], 10, 5, fimgL)
        b3=self.getBandGrid(iL[2], iL[3], 12, 6, fimgL)
        b4=self.getBandGrid(iL[3], iL[4], 14, 7, fimgL)
        b5=self.getBandGrid(iL[4], iL[5], 16, 8, fimgL)
        b6=self.getBandGrid(iL[5], iL[6], 18, 9, fimgL)
        b7=self.getBandGrid(iL[6], iL[7], 20, 10, fimgL)
        
        #create the 7 pooled bands by pooling the grids held in each b (i.e. b1, b2, ..., b7)
        clayerB1 = self.processC1(b1, 1)
        clayerB2 = self.processC1(b2, 2)
        clayerB3 = self.processC1(b3, 3)
        clayerB4 = self.processC1(b4, 4)
        clayerB5 = self.processC1(b5, 5)
        clayerB6 = self.processC1(b6, 6)
        clayerB7 = self.processC1(b7, 7)
        
        #return a list of pooled band grids
        allClayers=[clayerB1, clayerB2, clayerB3, clayerB4, clayerB5, clayerB6, clayerB7]
        return allClayers
    
    def prepareForClassification(self, globalN, imGData):
        """This function effectively pre-processes the raw image list ready for input into a classifier
        """
        
        #Do the first image in order to find the dimensions
        #This means processing one extra image in order to determine this
        #Although inefficient it guarantees that the result is correct
        #Given that 50000 images will be processed 1 additional is not such a big extra overhead
        
        #get a set of filter banks
        #only need to do this once
        gfb = self.buildGaborFilterBanks()
        
        j=0
        testImage = imGData[j]
        
        #convolve the filter banks with the image
        fimgL = self.process(testImage, gfb, conv=1)
        
        #get the pooled band grids
        cl = self.processAllBands(fimgL)
        
        #concatenate the results of the pooled band grids into a single vector
        first = 1
        for i in range(self.numberOfBands):
            for indexKey in cl[i].keys():
                if first == 1:
                    pcd = cl[i][indexKey]
                    first = 0
                else:
                    cd = cl[i][indexKey]
                    pcd = np.concatenate((pcd, cd))
    
        #the above code allows the determination of "theDim", which is hard to calculate ahead of time
        #this is the length of the 1-D vector required to hold the pooled band grids, all of which are a different 2-D size
        theDim = len(pcd)
        #print theDim
        
        #create the data matrix which will contain the input vectors for input into the classifiers
        #now we know "theDim" we can create the matrix ahead of time and populate it as we go
        imageDataMatrix = np.zeros((globalN, theDim), dtype=np.float32)

        
        #process the rest of the image list
        for j in range(globalN):
            
            #outward indication that processing is proceeding
            if (j + 1) % 500 == 0:
                print j + 1
                
            #get the next image for processing    
            testImage = imGData[j]
            
            #convolve the filter banks with the images
            fimgL = self.process(testImage, gfb, conv=1)
            
            #get the pooled band grids
            cl = self.processAllBands(fimgL)
            
            #concatenate the results of the pooled band grids into a single vector
            first = 1
            for i in range(self.numberOfBands):
                for indexKey in cl[i].keys():
                    #print indexKey
                    if first == 1:
                        pcd = cl[i][indexKey]
                        first = 0
                    else:
                        cd = cl[i][indexKey]
                        pcd = np.concatenate((pcd, cd))
    
            temp = pcd.reshape(pcd.shape[0])
        
            #populate the data matrix
            imageDataMatrix[j] = temp.copy()
            
        return imageDataMatrix
    
    def prepareOneForClassification(self, iM):
        """This function effectively pre-processes a single raw image ready for input into a classifier
        """
        
        #get a set of filter banks
        gfb = self.buildGaborFilterBanks()
        
        #convolve the filter banks with the image
        fimgL = self.process(iM, gfb, conv=1)
        
        #get the pooled band grids        
        cl = self.processAllBands(fimgL)
        
        #concatenate the results of the pooled band grids into a single vector
        first = 1
        for i in range(self.numberOfBands):
            for indexKey in cl[i].keys():
                if first == 1:
                    pcd = cl[i][indexKey]
                    first = 0
                else:
                    cd = cl[i][indexKey]
                    pcd = np.concatenate((pcd, cd))
    
        theDim = len(pcd)
        
        #create and populate the data matrix for the image
        imageDataMatrix = np.zeros((1, theDim), dtype=np.float32)
        temp = pcd.reshape(pcd.shape[0])
        imageDataMatrix[0] = temp.copy()
    
        return imageDataMatrix
    

    
    
#-------------------------------------------CLASS dimReduction-------------------------------------------


class dimReduction(object):
    """For expansion...class to contain dimensional reduction routines
    """
    
    def getPC(self, dataMatrix, numComponents = 25):
        """Apply pca to the data matrix. numComponents specifies the number of principal components to use
        """
        pca = decomposition.PCA(n_components=numComponents)
        
        #zero mean the data before utilizing pca
        dataMatrixZM = dataMatrix - dataMatrix.mean()
        pca.fit(dataMatrixZM)
        
        #return the transformed data matrix
        dataMatrixPCA = pca.transform(dataMatrixZM)
        return dataMatrixPCA
    
    
#-------------------------------------------CLASS dimReduction-------------------------------------------    
    
class vizFeatures(object):
    """A set of routines to assist in visualizing lots of images at once. This includes seeing the results of the Gabor
    filters and the band pooling.
    """
    
    def __init__(self):
        self.numberOfAngles = 8
    
    def plotBands(self, fimgL):
        """Plots the Gabor filter bands for feature extraction visualization purposes
        """
        
        #set up the axes - determining the number of rows and columns - number of columns is always the number of angles of
        #orientation of the Gabor filters
        rows = len(fimgL)/self.numberOfAngles + 1
        fig, axes = plt.subplots(nrows=rows, ncols=self.numberOfAngles, figsize=(20,20))
        
        #show the original image
        ax=axes[0][0]
        ax.imshow(testImage, cmap = plt.get_cmap('gray'))

        count = 0
        for r in range(0, rows):
            
            #turn the axes off
            for ax in axes[r][:]:
                ax.axis("off")
                
                #as there are no more images to "turn on" in row zero process for r>0
                if r > 0:
                    if count <= len(fimgL):
                        
                        #set the title and plot
                        ax.set_title(fimgL.keys()[count], fontsize=12)
                        ax.imshow(fimgL[fimgL.keys()[count]], cmap = plt.get_cmap('gray'))
                    count = count + 1
                    
    def plotC1Features(self, allClayers):
        """Plots the pooled band images for feature extraction vizualization
        """
        
        #determine the number of rows and the number of columns - number of columns is always the number of angles of
        #orientation of the Gabor filters
        rows=len(allClayers)+1
        fig, axes = plt.subplots(nrows=rows, ncols=self.numberOfAngles, figsize=(20,20))
        
        #plot the original image in the top left hand corner
        ax=axes[0][0]
        ax.imshow(testImage, cmap = plt.get_cmap('gray'))

        #turn the axes off for all columns in row one, except the the top left hand corner
        for ax in axes[0][1:]:
            ax.axis("off")

        for r in range(1, rows):
            #get indices for use in titling each image
            for count, ax in zip(range(self.numberOfAngles), axes[r][:]):
                
                #get the pooled filter band
                cl = allClayers[r-1]
                
                #set the title using the dictionary key
                ax.set_title(cl.keys()[count], fontsize=12)
                
                #figure out the image dimensions - the pooled filter bands are stored a vectors
                reshapeDim = np.sqrt(len(cl[cl.keys()[count]]))
                
                #display the pooled filter band
                ax.imshow(cl[cl.keys()[count]].reshape(reshapeDim, reshapeDim), cmap = plt.get_cmap('gray'))
        
    def displayOneImage(self, imageNo, bw=1):
        """Takes an image or image number and gets it from the training set and displays it, either in black and white or color
        """
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
        
        #if the imageNo is an int then get the image from store
        if type(imageNo) is int:
            
            #instantiate and input object
            gD = getData()
            
            #read in a single color image
            iM = gD.readOneImage(imageNo) 
            if bw == 1:
                ax.set_title(str(imageNo)+".png grayscale")
                
                #convert to grayscale
                iMG = rgb2gray(iM)
            else:
                ax.set_title(str(imageNo)+".png")
        
        #imageNo contains an actual image
        else:
            if bw == 1:
                iMG = imageNo
            else:
                iM = imageNo
        
        #display        
        if bw == 1:
            ax.imshow(iMG, cmap = plt.get_cmap('gray'))
        else:
            ax.imshow(iM)

            
    def displayOneTestImage(self, imageNo, bw=1):
        """Takes a raw test image number or test image and gets it from the test set and displays it, 
        either in black and white or color
        """
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
        
        #check to see if the imageNo contains just a plain int
        if type(imageNo) is int:
            
            #instantiate an input object
            gD = getData()
            
            #read in the test image in color from the test set
            iM = gD.readOneTestImage(imageNo) 
            if bw == 1:
                ax.set_title(str(imageNo)+".png grayscale")
                
                #convert to grayscale
                iMG = rgb2gray(iM)
            else:
                ax.set_title(str(imageNo)+".png")
        else:
            if bw == 1:
                iMG = imageNo
            else:
                iM = imageNo
        
        #display
        if bw == 1:
            ax.imshow(iMG, cmap = plt.get_cmap('gray'))
        else:
            ax.imshow(iM)


#-------------------------------------------CLASS objectClassification-------------------------------------------                
            
class objectClassification(object):
    """This class contains a bucket of functions relating to all aspects of training and running the classifiers
    """
    
    def __init__(self):
        """Some instance variables used to regularize some of the classifiers
        """
        
        self.C = 0.01
        self.minSamplesPerLeaf = 1
        self.maxDepth = 9
        self.numEstimators = 100
        self.learningRate = 1.5
    
    def accReport(self, indexName, model, xtrain, ytrain, xtest, ytest, doPrint=0):
        """Print the accuracy on the test and training dataset
        """
        
        training_accuracy = model.score(xtrain, ytrain)
        testing_accuracy = model.score(xtest, ytest)
        
        #sometimes don't do a print of the results, simply return the results
        if(doPrint == 1):
            print "Accuracy of %s  on training data: %0.2f" % (indexName, training_accuracy)
            print "Accuracy of %s on test data: %0.2f" % (indexName, testing_accuracy)
        
        results = (training_accuracy, testing_accuracy)
        return results
    
    def trainFullClassifier(self, imageData, imageLabels):
        """Originally hoped to set up a comprehensive array of classifiers to train. Owing to the problems in training
        - most notably the long time required to train, this function just used those that would train in a reasoable amount
        of time.
        This function used the full 50000 training set and the pre-determined 10000 test set - hence no cross validation routine
        """
        
        #simple tree classifier - min samples per leaf was the regularizer
        tree_clf = tree.DecisionTreeClassifier(min_samples_leaf = 4)
        
        #logistic regression - C value was the regularizer
        lr_clf = LogisticRegression(C = 0.1)
        
        #ada boost classifier - tried this on occasion and looked promising at one stage, but overall too slow
        ab_clf = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(min_samples_leaf = 4), 
                                    n_estimators = self.numEstimators, learning_rate = self.learningRate)
        
        #stochastic gradient descent - trained quickly but results not the best
        sgd_svm_clf = SGDClassifier(loss='log', n_iter=100, shuffle=True)
        
        #random forest - main stay classifier of the project - could set njobs = -1 to use multi-core processing
        #massive speedup
        rf_clf = RandomForestClassifier()
        
        #extra trees - also could set njobs = -1 to use multi-core processing
        #massive speedup
        et_clf = ExtraTreesClassifier()
        
        #create a dictionary of the classifiers to be tried
        classifiers = {'T':tree_clf, 'LR': lr_clf, 'AB' : ab_clf, 'SG' : sgd_svm_clf, 'RF' : rf_clf, 'ET' : et_clf}

        m = collections.OrderedDict()
                
        #train the models
        for  clfKey in classifiers.keys():
            clf = classifiers[clfKey]
            indexName = "mdl_" + clfKey
            
            #imageData was the full 50000 training set
            tC = clf.fit(imageData, imageLabels)
            m[indexName]  = tC 
        
        #return the models
        return m
    
    def trainWithClassifier(self, theName, description, clf, trainData, trainLabels, testData, testLabels):
        """This routine tests the classifier provided by "clf", and saves it all to disk
        The filename is designed to be unique, recording the date and time of the save
        Because the classifiers were hard to train it was important to save all the information about them once they had
        been successfully trained (as re-training was hard!).
        For each classifier store - a description
        - the classifier itself
        - the training set
        - the training lables
        - the test set
        - the test labels
        - the accuracy report
        - the cross tab information
        """
        
        #instantiate a getData object 
        data = getData()
        
        #get the system time for using as part of the filename for when the classifier is saved
        ts = time.time()
        
        #get the system date for using as part of the filename for when the classifier is saved
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
        
        #m was the output from trainFullClassifier()
        m = collections.OrderedDict()
        
    
        #for each classifier save the following to disk: a description, the training data, the test data, the training labels
        #the test labels, the accuracy report and the cross tab
        m[theName + "description"]=description
        m[theName + "clf"]=clf.fit(trainData, trainLabels)
        m[theName + "trainData"]=trainData.copy()
        m[theName + "trainLabels"]=trainLabels.copy()
        m[theName + "testData"]=testData.copy()
        m[theName + "testLabels"]=testLabels.copy()
        
        #test and record the classifier on the test data
        m[theName + "acc"]=self.accReport(theName, m[theName+"clf"], trainData, trainLabels, testData, testLabels, 1)
        m[theName + "ctab"]=pd.crosstab(testLabels, m[theName+"clf"].predict(testData),rownames = ["Actual"], colnames = ["Predicted"])
        
        #save the classifier
        data.saveData(m, theName+st, "results")
        
        #print the cross tab information
        print m[theName+"ctab"]
        return m
    
    def trainClassifiers(self, imageData, imageLabels) :
        """Originally hoped to set up a comprehensive array of classifiers to train. Owing to the problems in training
        - most notably the long time required to train, this function was largely unused
        This used cross validation over the training set
        Abandoned in light of the decision to use the entire training set and pre-determined test set to produce results that could
        be compared with those published in the literature
        """
        
        tree_clf = tree.DecisionTreeClassifier(max_depth = self.maxDepth, min_samples_leaf = self.minSamplesPerLeaf)      
        lr_clf = LogisticRegression(C = self.C)        
        ab_clf = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(), 
                                    n_estimators = self.numEstimators, learning_rate = self.learningRate)
                
        #set up a dictionary of classifiers
        classifiers = {'T':tree_clf, 'LR': lr_clf, 'AB' : ab_clf}
        m = collections.OrderedDict()       
        
        #used stratified shuffle split for cross validation
        sss = StratifiedShuffleSplit(imageLabels, 10, test_size = 0.1, random_state = 5)
        
        for  clfKey in classifiers.keys():
            i=1
            clf = classifiers[clfKey]
            
            #for each classifier perform cross validation
            for train_index, test_index in sss:
                
                #assign a name for the classifier
                indexName = "mdl_" + clfKey + str(i)
                i = i + 1
                X_train, X_test = imageData[train_index], imageData[test_index]
                y_train, y_test = imageLabels[train_index], imageLabels[test_index]
                
                #fit the model
                tC = clf.fit(X_train, y_train)
                
                m[indexName]  = tC
            
                #get the accuracy report
                m[indexName+"Acc"] = self.acc_report(indexName, tC, X_train, y_train, X_test, y_test, 1)
                m[indexName+"xtrain"] = X_train.copy()
                m[indexName+"xtest"] = X_test.copy()
                m[indexName+"ytrain"] = y_train.copy()
                m[indexName+"ytest"] = y_test.copy()
        
        #produce the cross tab resulst
        k=0
        for j in range(len(m.keys())):
            ctab = pd.crosstab(m[m.keys()[k+5]], m[m.keys()[k]].predict(m[m.keys()[k+3]]), 
                        rownames = ["Actual"], colnames = ["Predicted"])
            print "\n-----------------------------------------------------------------------"
            print m.keys()[k], "n"
            print ctab
            k = k + 6    
            
        #return all the trained classifiers, with their data and results    
        return m
    
    def trainTreeClassifiers(self, imageData, imageLabels) :
        """Originally hoped to set up a comprehensive array of classifiers to train. 
        Owing to the problems in training - most notably the long time required to train, this function was largely unused
        This used cross validation over the training set
        Abandoned in light of the decision to use the entire training set and pre-determined test set to produce results that could
        be compared with those published in the literature
        This function was being designed to just focus on tree classification models regularized by minimum samples per leaf
        """
        classifiers = collections.OrderedDict()
        
        for mSL in (1, 2, 3, 4, 5):
            indN = "treeAny" + ":" + str(mSL)
            classifiers[indN] = tree.DecisionTreeClassifier( min_samples_leaf = mSL)
                
        
        m = collections.OrderedDict()   
        
        #cross validation strategy
        sss = StratifiedShuffleSplit(imageLabels, 1, test_size = 0.5, random_state = 5)
        
        for  clfKey in classifiers.keys():
            i=1
            clf = classifiers[clfKey]
            for train_index, test_index in sss:
                indexName = "mdl_" + str(i) + clfKey 
                i = i + 1
                X_train, X_test = imageData[train_index], imageData[test_index]
                y_train, y_test = imageLabels[train_index], imageLabels[test_index]
                
                tC = clf.fit(X_train, y_train)
                m[indexName]  = tC
                m[indexName+"Acc"] = self.acc_report(indexName, tC, X_train, y_train, X_test, y_test, 1)
                m[indexName+"xtrain"] = X_train.copy()
                m[indexName+"xtest"] = X_test.copy()
                m[indexName+"ytrain"] = y_train.copy()
                m[indexName+"ytest"] = y_test.copy()
        
        k=0
        for j in range(len(m.keys())):
            ctab = pd.crosstab(m[m.keys()[k+5]], m[m.keys()[k]].predict(m[m.keys()[k+3]]), 
                        rownames = ["Actual"], colnames = ["Predicted"])
            print "\n-----------------------------------------------------------------------"
            print m.keys()[k], "n"
            print ctab
            k = k + 6    
        return m
    
    
    #---------------------------FUNCTIONS BELOW ARE FOR UTILIZING TRAINED MODELS---------------------------
    
    def runBinaryClassifierNoLabel(self, imageNo, model):
        """Badly named function. Takes an image and returns the pre-processed, feature extracted data matrix, which
        will be the input into the classifier
        """
        
        #instantiate required objects
        gD = getData()
        gfe = gaborFeatureExtraction()
        
        #get the test image
        iM = gD.readOneTestImage(imageNo)
        
        #pre process
        iMG = rgb2gray(iM)
        iMGE = exposure.equalize_hist(iMG)
        
        #extract the Gabor features
        imageDataMatrix = gfe.prepareOneForClassification(iMGE)
        
        #return the data matrix
        return imageDataMatrix
    
    def runBinaryClassifier(self, imageNo, model):
        """A similar function to the one above, but this time extracting the associated class label and returning it as well
        as the data matrix
        """
        
        #instantiate required objects
        gD = getData()
        gfe = gaborFeatureExtraction()
        
        #obtain the relevant label
        myLabels = gD.readTestLabels()
        imLabel = myLabels[myLabels['id']==imageNo]['class']
        
        #read the image 
        iM = gD.readOneTestImage(imageNo)
        
        #pre process
        iMG = rgb2gray(iM)
        iMGE = exposure.equalize_hist(iMG)
        
        #extract the Gabor features
        imageDataMatrix = gfe.prepareOneForClassification(iMGE)
        
        #return the data matrix and label in a list
        return [imageDataMatrix, imLabel]
             
    def ensureInt(self, n):
        """ Cast n as an np.array and then cast this as an int.
        """
        
        temp = np.array(n).astype(int)
        return(temp)
        
    def runAnimalOrMechanical(self, imageNo, model, suppressPrint):
        """ Run the two class classifier Animcal or Mechanical
        """
        
        #get the feature data matrix and class label for the image that is to be classified
        cInfo = self.runBinaryClassifier(imageNo, model)
        imageDataMatrix = cInfo[0]
        imLabel = cInfo[1]
        
        mechCA = {}
        mechCCA = ["animal", "mechanical"]
        
        #For demonstration purposed may want to print the results of this classifier
        if not suppressPrint:
            print "Known label for this image is : " , imLabel.values
            print "\n-----------FIRST LEVEL PREDICTION-------------" 
            print "This image is class: ", self.ensureInt(model.predict(imageDataMatrix[0]))
        
        #print the class label
        if model.predict(imageDataMatrix[0]) == 0:
            if not suppressPrint:
                print "Animal"
            res = "Animal"
            
        elif model.predict(imageDataMatrix[0]) == 1:
            if not suppressPrint:
                print "Mechanical"
            res = "Mechanical"
            
        else:
            print "unkown class position F"
            
        if not suppressPrint:
            
            #get the probabilities and then sort the probabilities from high to low
            for i in range(2):
                mechCA[mechCCA[i]] =  model.predict_proba(imageDataMatrix[0])[0][i]
            
            sortedHA = sorted(mechCA.items(), key=operator.itemgetter(1), reverse=True)
            
            #print the probabilities from high to low
            for i in range(2):
                    print sortedHA[i][0],"=","%0.3f"%(sortedHA[i][1]),
                    
        #return the input vector and the classification result of animcal or mechanical
        return [imageDataMatrix, res]
    
    def runAnimalOrMechanicalWithCutoff(self, imageNo, model, cutOff):
        """Run the binary classifier "animal" or "mechanical", but only allow classification when the probability is above 
        the user provided threshold "cutOff"
        """
        
        #Get the image and prepare it for input into the classifier
        cInfo = self.runBinaryClassifier(imageNo, model)
        imageDataMatrix = cInfo[0]
        imLabel = cInfo[1]
        
        #if the probability does not exceed the cutoff then return "none"
        res = "none"
        
        #determine the class prediction based on the probabilities
        if model.predict_proba(imageDataMatrix[0])[0][0] > model.predict_proba(imageDataMatrix[0])[0][1]:
            
            #does the class prediction probability exceed the cutoff?
            if model.predict_proba(imageDataMatrix[0])[0][0] > cutOff:
                
                #if yes then return the classification
                res = "Animal"
                
        elif model.predict_proba(imageDataMatrix[0])[0][1] > cutOff:
            res = "Mechanical"
                    
        return res
    
    def runSplitAnimals(self, imageNo, models, mA1, suppressPrint):
        """Part of the cascade. This runs the binary classifier that tries to separate
        bird, cat, dog from horse, deer or frog.
        """
        
        #Start the cascade by getting the result of the binary mechanical vs animal model
        classInfoA = self.runAnimalOrMechanical(imageNo, models, suppressPrint)
        imageDataMatrix = classInfoA[0]
        mOrA = classInfoA[1]
        
        mechCBCD = {}
        mechCCBCD = ["bird or cat or dog", "horse or deer or frog"]
        
        #Using the output of the binary classifier run the animal binary classifier
        if mOrA == "Animal":
            if not suppressPrint:
                print "\n\n----------SECOND LEVEL PREDICTION-------------"
                print "This image is class: ", self.ensureInt(mA1.predict(imageDataMatrix[0]))
            if mA1.predict(imageDataMatrix[0]) == 0:
                if not suppressPrint:
                    print "Bird or a Cat or a Dog"
                res = "BirdCatDog"
            elif mA1.predict(imageDataMatrix[0]) == 1: 
                if not suppressPrint:
                    print "Horse or a Deer or a Frog"
                res = "HorseDeerFrog"
            else:
                print "unkown class position E"
            if not suppressPrint:
                for i in range(2):
                    #get the probabilities
                    mechCBCD[mechCCBCD[i]] =  mA1.predict_proba(imageDataMatrix[0])[0][i]
                    
                #order the probabilities from high to low
                sortedHBCD = sorted(mechCBCD.items(), key=operator.itemgetter(1), reverse=True)
                
                for i in range(2):
                    #print the ordered probabilities
                    print sortedHBCD[i][0],"=","%0.3f"%(sortedHBCD[i][1]),
                    
        #if the binary classifier animal or mechanical thought the image was mechanical then bypass the animal classifier above and
        #just return "TruckorAutoorShiporPlane"
        elif mOrA == "Mechanical":
            res = "TruckorAutoorShiporPlane"
            
        #return the output from the animal or mechanical classifier, as well as the data matrix, as well as the output of the 
        #animal 2 class classifier
        return([mOrA, imageDataMatrix, res])

    def classifyHDF(self, imageNo, models, mA1, mA1b, mA1a, mA12, suppressPrint = 1):
        """This is the main workhorse of the cascade classification system - utilising 5 random forest classifiers
        """
        
        #At the top level obtain the outcome of the animal or mechanical classifier, and if animal the outcome of the 2 class
        #animal classifier - birdcatdog or horsedeerfrog
        classInfoA1b = self.runSplitAnimals(imageNo, models, mA1, suppressPrint)
        
        #animal or mechanical results
        mOrA = classInfoA1b[0]
        
        #we will utlize the input feature vector in the other classifiers
        imageDataMatrix = classInfoA1b[1]
        
        #birdcatdog or horsedeerfrog results
        hdfOrCDB = classInfoA1b[2]
        
        
        if mOrA == "Animal":
            if hdfOrCDB == "BirdCatDog":
                mechCDB = {}
                mechCCDB = ["bird", "cat", "dog"]
                
                #based on results so far we use the 3 class animal classifier to choose either a cat, a dog or a bird
                if not suppressPrint:
                    print "\n\n----------FINAL CLASS PREDICTION--------------"

                    print "This image is class: ", self.ensureInt(mA1b.predict(imageDataMatrix[0]))
                    
                if mA1b.predict(imageDataMatrix[0]) == 5:
                    if not suppressPrint:
                        print "Cat"
                    prediction ="cat"
                    
                elif mA1b.predict(imageDataMatrix[0]) == 6:
                    if not suppressPrint:
                       print "Dog"
                    prediction = "dog"
                    
                elif mA1b.predict(imageDataMatrix[0]) == 4:
                    if not suppressPrint:
                        print "Bird"
                    prediction = "bird"
                else:
                    print "unkown class position D"
                    
                if not suppressPrint:
                    
                    #report the probabilities in order from highest to lowest
                    for i in range(3):
                        mechCDB[mechCCDB[i]] =  mA1b.predict_proba(imageDataMatrix[0])[0][i]
                    sortedHCDB = sorted(mechCDB.items(), key=operator.itemgetter(1), reverse=True)
                    for i in range(3):
                        print sortedHCDB[i][0],"=","%0.3f"%(sortedHCDB[i][1]),
                
            elif hdfOrCDB == "HorseDeerFrog":
                
                mechHDF = {}
                mechCHDF = ["frog", "deer", "horse"]
                
                #based on results so far we use the 3 class animal classifier to choose either a frog, a deer or a horse
                if not suppressPrint:
                    print "\n\n---------FINAL CLASS PREDICTION---------------"
                    print "This image is class: ", self.ensureInt(mA1a.predict(imageDataMatrix[0]))
                    
                if mA1a.predict(imageDataMatrix[0]) == 0:
                    if not suppressPrint:
                        print "Frog"
                    prediction = "frog"
                    
                elif mA1a.predict(imageDataMatrix[0]) == 2:
                    if not suppressPrint:
                       print "Deer"
                    prediction = "deer"
                    
                elif mA1a.predict(imageDataMatrix[0]) == 7:
                    if not suppressPrint:
                       print "Horse"
                    prediction = "horse"
                    
                else:
                    print "unkown class position B"
                    
                if not suppressPrint:
                    #report the probabilities in order from highest to lowest

                    for i in range(3):
                        mechHDF[mechCHDF[i]] =  mA1a.predict_proba(imageDataMatrix[0])[0][i]
                    sortedHCDB = sorted(mechHDF.items(), key=operator.itemgetter(1), reverse=True)
                    for i in range(3):
                        print sortedHCDB[i][0],"=","%0.3f"%(sortedHCDB[i][1]),

                return prediction
        else:
            #if the animal or mechanical classifier result was mechanical then use the 4 class mechanical model to choose either
            #truck, automobile, ship or airplane
            mechD = {}
            mechC = ["truck", "automobile", "ship", "airplane"]
            
            if not suppressPrint:
                print "\n\n--------FINAL CLASS PREDICTION----------------"

                print "This image is class: ", self.ensureInt(mA12.predict(imageDataMatrix[0]))
            if mA12.predict(imageDataMatrix[0]) == 1:
                if not suppressPrint:
                    print "Truck"
                prediction ="truck"
            elif mA12.predict(imageDataMatrix[0]) == 3:
                if not suppressPrint:
                    print "Automobile"
                prediction = "automobile"
            elif mA12.predict(imageDataMatrix[0]) == 8:
                if not suppressPrint:
                    print "Ship"
                prediction = "ship"
            elif mA12.predict(imageDataMatrix[0]) == 9:
                if not suppressPrint:
                    print "Airplane"
                prediction = "airplane"
            else:
                print "unkown class position A"
                
            if not suppressPrint:
                #report the probabilities of each of the 4 outcomes in order from most probable to least probable
                for i in range(4):
                    mechD[mechC[i]] =  mA12.predict_proba(imageDataMatrix[0])[0][i]
                    
                sortedMechD = sorted(mechD.items(), key=operator.itemgetter(1), reverse=True)
                
                for i in range(4):
                    print sortedMechD[i][0],"=","%0.3f"%(sortedMechD[i][1]),
            
            #return the prediction of the cascade of classifiers
            return prediction

    def classHelper(self, theClass):  
        """This function is used by a number of other routines. Taking in a numeric class identifier and returning the string
        """
        
        if theClass == 0:
            return "frog"
        elif theClass == 1:
            return "truck"
        elif theClass == 2:
            return "deer"
        elif theClass == 3:
            return "automobile"
        elif theClass == 4:
            return "bird"
        elif theClass == 5:
            return "cat"
        elif theClass == 6:
            return "dog"
        elif theClass == 7:
            return "horse"
        elif theClass == 8:
            return "ship"
        elif theClass == 9:
            return "airplane"
                
    def runTenClass(self, imageNo, model):
        """Run the ten class classifier.
        """
        #get the feature vector and class label
        cInfo = self.runBinaryClassifier(imageNo, model)
        
        #this is the feature vector
        imageDataMatrix = cInfo[0]
        
        #this is the class label
        imLabel = cInfo[1]
        
        category = ["frog", "truck", "deer", "auto", "bird", "cat", "dog", "horse", "ship", "plane"]
        print "Known label for this image is :", imLabel.values
        
        print "\n\n---------------PREDICTIONS--------------------"
        print "This image is class: ", self.ensureInt(model.predict(imageDataMatrix[0]))
        
        #get the probabilities for all ten classes in sorted order from most probable to least probable
        anD = {}
        for i in range(10):
            anD[category[i]] =  model.predict_proba(imageDataMatrix[0])[0][i]
            
        sortedAnD = sorted(anD.items(), key=operator.itemgetter(1), reverse=True)
        for i in range(10):
            print sortedAnD[i][0],"=","%0.2f"%(sortedAnD[i][1]),
            
        print "\n"
        theClass = model.predict(imageDataMatrix[0])
        resClass = self.classHelper(theClass)
        print resClass
            
    def runTenClassWithCutoff(self, imageNo, model, cutOff):
        """Run the ten class classifier using a cutoff for the probabilities, i.e. don't classify unless the probability 
        exceeds the cutoff. Print interactively - for demonstration purposes
        """
        
        #get the feature vector and class label
        cInfo = self.runBinaryClassifier(imageNo, model)
        
        #feature vector
        imageDataMatrix = cInfo[0]
        
        #class label
        imLabel = cInfo[1]
        
        
        category = ["frog", "truck", "deer", "auto", "bird", "cat", "dog", "horse", "ship", "plane"]
        print "Known label for this image is :", imLabel.values
        
        print "\n\n---------------PREDICTIONS--------------------"
        
        #get the probabilities for all ten classes in sorted order from most probable to least probable
        anD = {}
        for i in range(10):
            anD[category[i]] =  model.predict_proba(imageDataMatrix[0])[0][i]
            
        sortedAnD = sorted(anD.items(), key=operator.itemgetter(1), reverse=True)
        for i in range(10):
            print sortedAnD[i][0],"=","%0.2f"%(sortedAnD[i][1]),
            
        print "\n"
        
        
        #Find the maxiumum probabilitiy
        max = -np.Inf
        for i in range(10):
            if model.predict_proba(imageDataMatrix[0])[0][i] > max:
                max = model.predict_proba(imageDataMatrix[0])[0][i]
                
        #only report and use the classification if it exceeds the cutoff        
        if max > cutOff:
            print "This image is class: ", self.ensureInt(model.predict(imageDataMatrix[0]))        
            theClass = model.predict(imageDataMatrix[0])
            resClass = self.classHelper(theClass)
            print resClass
            
        #best probability does not exceed the cutoff so do not classify    
        else:
            print "No dominant probability ( >", cutOff, "). No classification undertaken"
            
    def getClassWithCutoff(self, imageNo, model, cutOff):
        """Run the ten class classifier using a cutoff for the probabilities, i.e. don't classify unless the probability 
        exceeds the cutoff. Same as above but no interactive printing
        """
        
        #get the feature vector and class label
        cInfo = self.runBinaryClassifier(imageNo, model)
        
        #feature vector
        imageDataMatrix = cInfo[0]
        
        #class label
        imLabel = cInfo[1]
        
        #find the maximum probability
        max = -np.Inf
        for i in range(10):
            if model.predict_proba(imageDataMatrix[0])[0][i] > max:
                max = model.predict_proba(imageDataMatrix[0])[0][i]
                
        #probability exceeds the cutoff so make the prediction
        if max > cutOff:
                    
            theClass = model.predict(imageDataMatrix[0])
            resClass = self.classHelper(theClass)
            return resClass
        
        #probability does not exceed the cutoff
        else:
            return "none"
