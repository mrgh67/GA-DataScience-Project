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

class getData(object):
    "handles all the input required, and initial data munging"
    
    def __init__(self, machine="laptop", announce = 0):
        self.machine = machine
        if announce == 1:
            print "You are working on", self.machine
        if(self.machine == "laptop"):
            self.labelPath = "/home/vagrant/Downloads/GAPROJECT/TRAIN/"
            self.filePath = "/home/vagrant/Downloads/GAPROJECT/TRAIN/train/"
            self.testFilePath = "/home/vagrant/Downloads/GAPROJECT/TEST/"
            self.resultsCPath = "/home/vagrant/Downloads/GAPROJECT/RESULTS/CASCADE/"
            self.resultsTPath = "/home/vagrant/Downloads/GAPROJECT/RESULTS/TENCLASS/"
            self.picklePath = "/home/vagrant/Downloads/GAPROJECT/PICKLE/"
            
        elif(self.machine == "desktop"):
            self.labelPath = "/Users/mrgh/Desktop/GAP/TRAIN/"
            self.filePath = "/Users/mrgh/Desktop/GAP/TRAIN/train/"
            self.testFilePath = "/Users/mrgh/Desktop/GAP/TEST/"
            self.resultsCPath = "/Users/mrgh/Desktop/GAP/RESULTS/CASCADE/"
            self.resultsTPath = "/Users/mrgh/Desktop/GAP/RESULTS/TENCLASS/"
            self.picklePath = "/Users/mrgh/GA_PYTHON_2/"
            
        elif(self.machine == "wdesktop"):
            self.labelPath = "/Users/markholt/Desktop/TRAIN/"
            self.filePath = "/Users/markholt/Desktop/TRAIN/train/"
            self.testFilePath = "/Users/markholt/Desktop/TEST/"
            self.resultsPath = "/Users/markholt/Desktop/RESULTS/"
            
    def getPath(self, p):
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
        myLabels = pd.read_csv(self.labelPath + "SmallObjectImages_Labels.csv")
        return myLabels
    
    def readTestLabels(self):
        myLabels = pd.read_csv(self.testFilePath + "test_labels.csv")
        return myLabels
    
    def readImages(self, tNumber):
        imageList = []
        for i in range(1, tNumber + 1):
            fileName = self.filePath + str(i) + ".png"
            im1 = mpimg.imread(fileName)
            imageList.append(im1)   
        return imageList
    
    def readTestImages(self, tNumber):
        imageList = []
        for i in range(1, tNumber + 1):
            fileName = self.testFilePath + str(i) + ".png"
            im1 = mpimg.imread(fileName)
            imageList.append(im1)   
        return imageList
    
    def readListOfImages(self, imList):
        imageList = []
        for i in imList:
            fileName = self.filePath + str(i) + ".png"
            im1 = mpimg.imread(fileName)
            imageList.append(im1)   
        return imageList
                   
    def readOneImage(self, imageNumber):
        fileName = self.filePath + str(imageNumber) + ".png"
        im1 = mpimg.imread(fileName)
        return im1
    
    def readOneTestImage(self, imageNumber):
        fileName = self.testFilePath + str(imageNumber) + ".png"
        im1 = mpimg.imread(fileName)
        return im1
    
    def readImagesOfType(self, tType):
        myLabels = self.readLabels()
        myLabels = myLabels.sort_index(by='label').copy()
        imagesOfTypeDf = pd.DataFrame(index=range(5000), columns=["image_no", "label"])
        imagesOfTypeDf["label"] = tType
        imagesOfTypeDf["image_no"] = myLabels[myLabels['label']==tType]["id"].values
        return imagesOfTypeDf.sort_index(by="image_no")
    
    def pickRandomSet(self, req, numClasses, classifier):
        imLabList = []
        imList = []
        
        for klas in req:
            reqIm = self.readImagesOfType(klas)
            #mask is a list of image file name numbers e.g. 1.png
            mask = reqIm['image_no'].values
            ranIm = np.random.choice(mask,req[klas])
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
        if x == "frog" or x == "deer" or x == "bird" or x == "cat" or x == "dog" or x == "horse":
            return 0
        else:
            return 1
        
    def transformLabelsForTwoClassesA1(self, x):
        if x == "frog" or x == "deer" or x == "horse":
            return 0
        elif x == "bird" or x == "cat" or x == "dog":
            return 1
        
    def transformLabelsForThreeClassesA1b(self, x):
        if x == "frog":
            return 0
        elif x == "deer":
            return 1
        elif x == "horse":
            return 2
        
    def transformLabelsForThreeClassesA1a(self, x):
        if x == "cat":
            return 0
        elif x == "dog":
            return 1
        elif x == "bird":
            return 2
        
    def transformLabelsForFourClassesA2(self, x):
        if x == "truck":
            return 0
        elif x == "automobile":
            return 1
        elif x == "ship":
            return 2
        elif x == "airplane":
            return 3
                
    def saveData(self, data, fileName, path):
        if path == "results":
            thePath = self.resultsPath
        else:
            thePath = path
        with open(thePath + fileName + ".pik", "wb") as f:
            pickle.dump(data, f, -1)
                        
    def retrieveData(self, fileName, path):
        if path == "results":
            thePath = self.resultsPath
        else: 
            thePath = path
        with open(thePath + fileName, "rb") as f:
            data = pickle.load(f)
        return data
    
    
class preProcess(object):
    "provides a set of basic pre-processing routines"
    
    def convertToGray(self, imageList):
        #converts a color image to grayscale
        
        imageGrayList = []
        for image in imageList:
            imG = rgb2gray(image)
            imageGrayList.append(imG)
        return imageGrayList
    
    def whitenGrayImage(self, imageList):
        #whitens a grayscale image - meaning zero mean unit standard deviation
        
        imageWList = []
        for image in imageList:
            imageN = (image - image.mean()) / image.std()
            imageWList.append(imageN)
        return imageWList
    
    def globalHistEq(self, imageList):
        #performs histogram equalization on the image
        
        gHEList = []
        for image in imageList:
            imageE = exposure.equalize_hist(image)
            gHEList.append(imageE)
        return gHEList
    
    def colorSplit(self, imageList, numChannels=3):
        #splits up a color image into its constituent red, green, and blue components.
        #by setting numChannels = 4 will handle an image with an alpha channel
        #any alpha channel is discarded
        
        rList = []
        gList = []
        bList = []
        colorList = collections.OrderedDict()
        for image in imageList:
            if numChannels == 3:
                b, g, r = cv2.split(image)
            elif numChannels == 4:
                b, g, r, a = cv2.split(image)
            bList.append(b)
            gList.append(g)
            rList.append(r)
        colorList["red"] = rList
        colorList["green"] = gList
        colorList["blue"] = bList
        return(colorList)
    
class featureExtraction(object):
    "undertakes other, more generic, feature extraction"
    
    def cannyFilter(self, imageList, tSigma = 2.5):
        imageCannyList = []
        for image in imageList:
            imageC = filter.canny(image, sigma=tSigma)
            imageCannyList.append(imagec)
        return imageCannyList    

class gaborFeatureExtraction(object):
    "undertakes all feature extraction for using Gabor Filter Banks"
    
    def __init__(self):
        self.imageDim = 32
        self.numberOfBands = 7
        self.numberOfBandsByTwo = 4
        self.numberOfAngles = 8
        
    def power(self, image, kernel):
        # Normalize images for better comparison.
        image = (image - image.mean()) / image.std()
        return np.sqrt(nd.convolve(image, np.real(kernel), mode='wrap')**2 +
                       nd.convolve(image, np.imag(kernel), mode='wrap')**2)
    
        
    def buildGaborFilterBanks(self, useOpenCV=1):
        #this produces a set of Gabor filter kernels
        
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
        #This function performs the convolution of the filter with the image
        
        #Maintaining order is useful
        fimgL = collections.OrderedDict()
        
        for idx, kernel in filters.items():
            if(conv == 1):
                #use the opencv filter routine for the convolution
                #print "opencv filter"
                fimg = cv2.filter2D(img, -1, kernel)
            else:
                #or use the skimage image convolve routine
                #print "skimage convolve"
                #fimg = nd.convolve(img, kernel, mode='wrap')
                fimg = self.power(img, kernel)
            #accumulate the filtered images    
            fimgL[idx] = fimg
        return fimgL
    
    def getC1(self, neighborhoodSize, overlap, indx, fimgL):
        #pool the results of Gabor filtering into a small square grid, the size of which is determineed by "neighborhoodsize"
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
        #for each band of frequencies accumulate a grid - using "getC1"
        bandGrid = collections.OrderedDict()
        
        #a band consists of two sets of closely related Gabor filters
        for i in range(a, b):
            indx = fimgL.keys()[i]
            bandGrid[indx] = self.getC1(neighborhoodSize, overlap, indx, fimgL)
        return bandGrid   

    def maxOfGrids(self, grida, gridb):
        #using a pair of bands, from which a pair of grids is derived, collpase the two grids into a single grid
        
        dim = len(grida)
        gridF = np.zeros(dim)
        
        #sum the two band grids recording the maximum output
        for x in range(dim):
            if grida[x] > gridb[x]:
                gridF[x] = grida[x]
            else:
                gridF[x] = gridb[x]
                
        #return the single summed grid        
        return gridF
    
    def processC1(self, b, bandNumber):
        clayer = collections.OrderedDict()
        for i in range(self.numberOfAngles):
            idxName = b.keys()[i][0:2] + "Band" + str(bandNumber)
            #print idxName, b.keys()[i], b.keys()[i+self.numberOfAngles]
            clayer[idxName] = self.maxOfGrids(b[b.keys()[i]], b[b.keys()[i+self.numberOfAngles]])
    
        return clayer
        
    def processAllBands(self, fimgL):
        
        iL = []
        for i in range(self.numberOfAngles):
            nL = i * self.numberOfAngles * 2
            iL.append(nL)
        #print iL
        #print len(iL)
        
            
        b1=self.getBandGrid(iL[0], iL[1], 8, 4, fimgL)
        b2=self.getBandGrid(iL[1], iL[2], 10, 5, fimgL)
        b3=self.getBandGrid(iL[2], iL[3], 12, 6, fimgL)
        b4=self.getBandGrid(iL[3], iL[4], 14, 7, fimgL)
        b5=self.getBandGrid(iL[4], iL[5], 16, 8, fimgL)
        b6=self.getBandGrid(iL[5], iL[6], 18, 9, fimgL)
        b7=self.getBandGrid(iL[6], iL[7], 20, 10, fimgL)
        
        clayerB1 = self.processC1(b1, 1)
        clayerB2 = self.processC1(b2, 2)
        clayerB3 = self.processC1(b3, 3)
        clayerB4 = self.processC1(b4, 4)
        clayerB5 = self.processC1(b5, 5)
        clayerB6 = self.processC1(b6, 6)
        clayerB7 = self.processC1(b7, 7)
        
        allClayers=[clayerB1, clayerB2, clayerB3, clayerB4, clayerB5, clayerB6, clayerB7]
        return allClayers
    
    def prepareForClassification(self, globalN, imGData):
        gfb = self.buildGaborFilterBanks()
        
        j=0
        testImage = imGData[j]
        fimgL = self.process(testImage, gfb, conv=1)
        cl = self.processAllBands(fimgL)
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
        print theDim
        imageDataMatrix = np.zeros((globalN, theDim), dtype=np.float32)

        for j in range(globalN):
            if (j + 1) % 500 == 0:
                print j + 1
            testImage = imGData[j]
            fimgL = self.process(testImage, gfb, conv=1)
            cl = self.processAllBands(fimgL)
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
            imageDataMatrix[j] = temp.copy()
            
        return imageDataMatrix
    
    def prepareOneForClassification(self, iM):
        gfb = self.buildGaborFilterBanks()
        fimgL = self.process(iM, gfb, conv=1)
        cl = self.processAllBands(fimgL)
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
        imageDataMatrix = np.zeros((1, theDim), dtype=np.float32)
        temp = pcd.reshape(pcd.shape[0])
        imageDataMatrix[0] = temp.copy()
    
        return imageDataMatrix
    
class dimReduction(object):
    
    def getPC(self, dataMatrix, numComponents = 25):
        pca = decomposition.PCA(n_components=numComponents)
        dataMatrixZM = dataMatrix - dataMatrix.mean()
        pca.fit(dataMatrixZM)
        dataMatrixPCA = pca.transform(dataMatrixZM)
        return dataMatrixPCA
    
class vizFeatures(object):
    def __init__(self):
        self.numberOfAngles = 8
    
    def plotBands(self, fimgL):
        rows = len(fimgL)/self.numberOfAngles + 1
        fig, axes = plt.subplots(nrows=rows, ncols=self.numberOfAngles, figsize=(20,20))
        ax=axes[0][0]
        ax.imshow(testImage, cmap = plt.get_cmap('gray'))

        count = 0
        for r in range(0, rows):
            for ax in axes[r][:]:
                ax.axis("off")
                if r > 0:
                    if count <= len(fimgL):
                        ax.set_title(fimgL.keys()[count], fontsize=12)
                        ax.imshow(fimgL[fimgL.keys()[count]], cmap = plt.get_cmap('gray'))
                    count = count + 1
                    
    def plotC1Features(self, allClayers):
        rows=len(allClayers)+1
        fig, axes = plt.subplots(nrows=rows, ncols=self.numberOfAngles, figsize=(20,20))
        ax=axes[0][0]
        ax.imshow(testImage, cmap = plt.get_cmap('gray'))

        for ax in axes[0][1:]:
            ax.axis("off")

        for r in range(1, rows):
            for count, ax in zip(range(self.numberOfAngles), axes[r][:]):
                #print count
                #ax.axis("off")
                cl = allClayers[r-1]
                ax.set_title(cl.keys()[count], fontsize=12)
                reshapeDim = np.sqrt(len(cl[cl.keys()[count]]))
                ax.imshow(cl[cl.keys()[count]].reshape(reshapeDim, reshapeDim), cmap = plt.get_cmap('gray'))
                
    def plotImageAndHistogram(self, img, axes, bins=256):
    
        #img = img_as_float(img)
        ax_img, ax_hist = axes
        #ax_cdf = ax_hist.twinx()

        # Display image
        ax_img.imshow(img, cmap=plt.cm.gray)
        ax_img.set_axis_off()

        # Display histogram
        #ax_hist.hist(img.ravel(), bins=bins, histtype='step', color='black')
        #ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        #ax_hist.set_xlabel('Pixel intensity')
        #ax_hist.set_xlim(0, 1)
        #ax_hist.set_yticks([])

        # Display cumulative distribution
        #img_cdf, bins = exposure.cumulative_distribution(img, bins)
        #ax_cdf.plot(bins, img_cdf, 'r')
        #ax_cdf.set_yticks([])
        
    def displayOneImage(self, imageNo, bw=1):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
        
        if type(imageNo) is int:
            gD = getData()
            iM = gD.readOneImage(imageNo) 
            if bw == 1:
                ax.set_title(str(imageNo)+".png grayscale")
                iMG = rgb2gray(iM)
            else:
                ax.set_title(str(imageNo)+".png")
        else:
            if bw == 1:
                iMG = imageNo
            else:
                iM = imageNo
        
        if bw == 1:
            ax.imshow(iMG, cmap = plt.get_cmap('gray'))
        else:
            ax.imshow(iM)

            
    def displayOneTestImage(self, imageNo, bw=1):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
        
        if type(imageNo) is int:
            gD = getData()
            iM = gD.readOneTestImage(imageNo) 
            if bw == 1:
                ax.set_title(str(imageNo)+".png grayscale")
                iMG = rgb2gray(iM)
            else:
                ax.set_title(str(imageNo)+".png")
        else:
            if bw == 1:
                iMG = imageNo
            else:
                iM = imageNo
        
        if bw == 1:
            ax.imshow(iMG, cmap = plt.get_cmap('gray'))
        else:
            ax.imshow(iM)


class objectClassification(object):
    
    def __init__(self):
        self.C = 0.01
        self.minSamplesPerLeaf = 1
        self.maxDepth = 9
        self.numEstimators = 100
        self.learningRate = 1.5
    
    def accReport(self, indexName, model, xtrain, ytrain, xtest, ytest, doPrint=0):
        #Print the accuracy on the test and training dataset
        training_accuracy = model.score(xtrain, ytrain)
        testing_accuracy = model.score(xtest, ytest)
        
        if(doPrint == 1):
            print "Accuracy of %s  on training data: %0.2f" % (indexName, training_accuracy)
            print "Accuracy of %s on test data: %0.2f" % (indexName, testing_accuracy)
        
        results = (training_accuracy, testing_accuracy)
        return results
    
    def trainFullClassifier(self, imageData, imageLabels) :
        tree_clf = tree.DecisionTreeClassifier(min_samples_leaf = 4)
        lr_clf = LogisticRegression(C = 0.1)
        ab_clf = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(min_samples_leaf = 4), 
                                    n_estimators = self.numEstimators, learning_rate = self.learningRate)
        sgd_svm_clf = SGDClassifier(loss='log', n_iter=100, shuffle=True)
        rf_clf = RandomForestClassifier()
        et_clf = ExtraTreesClassifier()
        
        #classifiers = {'T':tree_clf, 'LR': lr_clf, 'AB' : ab_clf}
        classifiers = {'AB' : ab_clf}

        m = collections.OrderedDict()
                
        for  clfKey in classifiers.keys():
            clf = classifiers[clfKey]
            indexName = "mdl_" + clfKey
            tC = clf.fit(imageData, imageLabels)
            m[indexName]  = tC 
        return m
    
    def trainWithClassifier(self, theName, description, clf, trainData, trainLabels, testData, testLabels):
        data = getData()
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
        m = collections.OrderedDict()
        m[theName + "description"]=description
        m[theName + "clf"]=clf.fit(trainData, trainLabels)
        m[theName + "trainData"]=trainData.copy()
        m[theName + "trainLabels"]=trainLabels.copy()
        m[theName + "testData"]=testData.copy()
        m[theName + "testLabels"]=testLabels.copy()
        m[theName + "acc"]=self.accReport(theName, m[theName+"clf"], trainData, trainLabels, testData, testLabels, 1)
        m[theName + "ctab"]=pd.crosstab(testLabels, m[theName+"clf"].predict(testData),rownames = ["Actual"], colnames = ["Predicted"])
        data.saveData(m, theName+st, "results")
        print m[theName+"ctab"]
        return m
    
    def trainClassifiers(self, imageData, imageLabels) :
        
        tree_clf = tree.DecisionTreeClassifier(max_depth = self.maxDepth, min_samples_leaf = self.minSamplesPerLeaf)      
        lr_clf = LogisticRegression(C = self.C)        
        ab_clf = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(), 
                                    n_estimators = self.numEstimators, learning_rate = self.learningRate)
                
        classifiers = {'T':tree_clf, 'LR': lr_clf, 'AB' : ab_clf}
        m = collections.OrderedDict()       
        sss = StratifiedShuffleSplit(imageLabels, 10, test_size = 0.1, random_state = 5)
        
        for  clfKey in classifiers.keys():
            i=1
            clf = classifiers[clfKey]
            for train_index, test_index in sss:
                indexName = "mdl_" + clfKey + str(i)
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
    
    def trainTreeClassifiers(self, imageData, imageLabels) :
        
        classifiers = collections.OrderedDict()
        #for mD in (3, 4, 5, 6, 7, 8, 9, 10):
        for mSL in (1, 2, 3, 4, 5):
            indN = "treeAny" + ":" + str(mSL)
            classifiers[indN] = tree.DecisionTreeClassifier( min_samples_leaf = mSL)
                
        #classifiers = {'T':tree_clf, 'LR': lr_clf, 'AB' : ab_clf}
        
        m = collections.OrderedDict()       
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
    
    def runBinaryClassifierNoLabel(self, imageNo, model):
        gD = getData()
        gfe = gaborFeatureExtraction()
        iM = gD.readOneTestImage(imageNo)
        iMG = rgb2gray(iM)
        iMGE = exposure.equalize_hist(iMG)
        imageDataMatrix = gfe.prepareOneForClassification(iMGE)
        return imageDataMatrix
    
    def runBinaryClassifier(self, imageNo, model):
        gD = getData()
        gfe = gaborFeatureExtraction()
        myLabels = gD.readTestLabels()
        imLabel = myLabels[myLabels['id']==imageNo]['class']
        iM = gD.readOneTestImage(imageNo)
        iMG = rgb2gray(iM)
        iMGE = exposure.equalize_hist(iMG)
        imageDataMatrix = gfe.prepareOneForClassification(iMGE)
        return [imageDataMatrix, imLabel]
             
    def ensureInt(self, n):
        temp = np.array(n).astype(int)
        return(temp)
        
    def runAnimalOrMechanical(self, imageNo, model, suppressPrint):
        cInfo = self.runBinaryClassifier(imageNo, model)
        imageDataMatrix = cInfo[0]
        imLabel = cInfo[1]
        
        mechCA = {}
        mechCCA = ["animal", "mechanical"]
        
        if not suppressPrint:
            print "Known label for this image is : " , imLabel.values
            print "\n-----------FIRST LEVEL PREDICTION-------------" 

            print "This image is class: ", self.ensureInt(model.predict(imageDataMatrix[0]))
            
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
            for i in range(2):
                    mechCA[mechCCA[i]] =  model.predict_proba(imageDataMatrix[0])[0][i]
                    sortedHA = sorted(mechCA.items(), key=operator.itemgetter(1), reverse=True)
            for i in range(2):
                    print sortedHA[i][0],"=","%0.3f"%(sortedHA[i][1]),
                    
        return [imageDataMatrix, res]
    
    def runSplitAnimals(self, imageNo, models, mA1, suppressPrint):
        classInfoA = self.runAnimalOrMechanical(imageNo, models, suppressPrint)
        imageDataMatrix = classInfoA[0]
        mOrA = classInfoA[1]
        mechCBCD = {}
        mechCCBCD = ["bird or cat or dog", "horse or deer or frog"]
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
                    mechCBCD[mechCCBCD[i]] =  mA1.predict_proba(imageDataMatrix[0])[0][i]
                sortedHBCD = sorted(mechCBCD.items(), key=operator.itemgetter(1), reverse=True)
                for i in range(2):
                    print sortedHBCD[i][0],"=","%0.3f"%(sortedHBCD[i][1]),
                    
        elif mOrA == "Mechanical":
            res = "TruckorAutoorShiporPlane"
        return([mOrA, imageDataMatrix, res])

    def classifyHDF(self, imageNo, models, mA1, mA1b, mA1a, mA12, suppressPrint = 1):
        classInfoA1b = self.runSplitAnimals(imageNo, models, mA1, suppressPrint)
        mOrA = classInfoA1b[0]
        imageDataMatrix = classInfoA1b[1]
        hdfOrCDB = classInfoA1b[2]
        if mOrA == "Animal":
            if hdfOrCDB == "BirdCatDog":
                mechCDB = {}
                mechCCDB = ["bird", "cat", "dog"]
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
                    for i in range(3):
                        mechCDB[mechCCDB[i]] =  mA1b.predict_proba(imageDataMatrix[0])[0][i]
                    sortedHCDB = sorted(mechCDB.items(), key=operator.itemgetter(1), reverse=True)
                    for i in range(3):
                        print sortedHCDB[i][0],"=","%0.3f"%(sortedHCDB[i][1]),
                
            elif hdfOrCDB == "HorseDeerFrog":
                
                mechHDF = {}
                mechCHDF = ["frog", "deer", "horse"]
                
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
                    for i in range(3):
                        mechHDF[mechCHDF[i]] =  mA1a.predict_proba(imageDataMatrix[0])[0][i]
                    sortedHCDB = sorted(mechHDF.items(), key=operator.itemgetter(1), reverse=True)
                    for i in range(3):
                        print sortedHCDB[i][0],"=","%0.3f"%(sortedHCDB[i][1]),

                return prediction
        else:
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
                for i in range(4):
                    mechD[mechC[i]] =  mA12.predict_proba(imageDataMatrix[0])[0][i]
                    
                sortedMechD = sorted(mechD.items(), key=operator.itemgetter(1), reverse=True)
                
                for i in range(4):
                    print sortedMechD[i][0],"=","%0.3f"%(sortedMechD[i][1]),
                    
            return prediction

                
    def runTenClass(self, imageNo, model):
        cInfo = self.runBinaryClassifier(imageNo, model)
        imageDataMatrix = cInfo[0]
        imLabel = cInfo[1]
        category = ["frog", "truck", "deer", "auto", "bird", "cat", "dog", "horse", "ship", "plane"]
        print "Known label for this image is :", imLabel.values
        
        print "\n\n---------------PREDICTIONS--------------------"
        print "This image is class: ", self.ensureInt(model.predict(imageDataMatrix[0]))
        
        anD = {}
        for i in range(10):
            anD[category[i]] =  model.predict_proba(imageDataMatrix[0])[0][i]
            #print category[i],"%0.3f" % (model.predict_proba(imageDataMatrix[0])[0][i]), 
            
        sortedAnD = sorted(anD.items(), key=operator.itemgetter(1), reverse=True)
        for i in range(10):
            print sortedAnD[i][0],"=","%0.2f"%(sortedAnD[i][1]),
            
        print "\n"
        theClass = model.predict(imageDataMatrix[0])
        if theClass == 0:
            print "frog"
        elif theClass == 1:
            print "truck"
        elif theClass == 2:
            print "deer"
        elif theClass == 3:
            print "automobile"
        elif theClass == 4:
            print "bird"
        elif theClass == 5:
            print "cat"
        elif theClass == 6:
            print "dog"
        elif theClass == 7:
            print "horse"
        elif theClass == 8:
            print "ship"
        elif theClass == 9:
            print "airplane"
            
    def runTenClassWithCutoff(self, imageNo, model, cutOff):
        cInfo = self.runBinaryClassifier(imageNo, model)
        imageDataMatrix = cInfo[0]
        imLabel = cInfo[1]
        category = ["frog", "truck", "deer", "auto", "bird", "cat", "dog", "horse", "ship", "plane"]
        print "Known label for this image is :", imLabel.values
        
        print "\n\n---------------PREDICTIONS--------------------"
        
        anD = {}
        for i in range(10):
            anD[category[i]] =  model.predict_proba(imageDataMatrix[0])[0][i]
            #print category[i],"%0.3f" % (model.predict_proba(imageDataMatrix[0])[0][i]), 
            
        sortedAnD = sorted(anD.items(), key=operator.itemgetter(1), reverse=True)
        for i in range(10):
            print sortedAnD[i][0],"=","%0.2f"%(sortedAnD[i][1]),
            
        print "\n"
        
        max = -np.Inf
        for i in range(10):
            if model.predict_proba(imageDataMatrix[0])[0][i] > max:
                max = model.predict_proba(imageDataMatrix[0])[0][i]
        if max > cutOff:
            print "This image is class: ", self.ensureInt(model.predict(imageDataMatrix[0]))
            #for i in range(10):
             #   print category[i],"%0.3f" % (model.predict_proba(imageDataMatrix[0])[0][i]),
            #print "\n"
        
            theClass = model.predict(imageDataMatrix[0])
            if theClass == 0:
                print "frog"
            elif theClass == 1:
                print "truck"
            elif theClass == 2:
                print "deer"
            elif theClass == 3:
                print "automobile"
            elif theClass == 4:
                print "bird"
            elif theClass == 5:
                print "cat"
            elif theClass == 6:
                print "dog"
            elif theClass == 7:
                print "horse"
            elif theClass == 8:
                print "ship"
            elif theClass == 9:
                print "airplane"
        else:
            print "No dominant probability ( >", cutOff, "). No classification undertaken"
