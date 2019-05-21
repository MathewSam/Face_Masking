import os

import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import match_template
import cv2

class DataHandler:
    def __init__(self,folder_name):
        '''
        Initializes the data handler to scan the dataset to extract locations where the person is present.
        Args:
            self: pointer pointing to current instance of the class
            folder_name: name of folder containing the images
        '''
        assert isinstance(folder_name,str),"Please enter valid name for location of images. Input must be a string input"
        #assert os._isdir(folder_name),"Please enter valid name for location of images. Input must be name of folder with images"

        self.root = folder_name
        self.filenames = os.listdir(folder_name)
        self.filenames.sort()

        image = cv2.imread(self.root + self.filenames[0])
        self.r = list(cv2.selectROI(image))
        self.template = image[int(self.r[1]):int(self.r[1]+self.r[3]), int(self.r[0]):int(self.r[0]+self.r[2])]
        cv2.imshow("Template", self.template)
        self.template = self.template[:,:,::-1]#Convert BGR to RGB
        cv2.waitKey(0)

    def iterative_search(self):
        '''
        Function that carries out masking of the image
        Args:
            self: pointer to the current instance of the class
        '''
        for index,file_name in enumerate(self.filenames):
            if index==0:
                continue
            else:
                image = plt.imread(self.root + file_name)
                temp = image[int(self.r[1]-100):int(self.r[1]+self.r[3]+100), int(self.r[0]-100):int(self.r[0]+self.r[2]+100)]
                masked = image.copy()
                result = match_template(temp, self.template)
                ij = np.unravel_index(np.argmax(result), result.shape)
                #import pdb;pdb.set_trace()
                self.r[1] = self.r[1]-100 + ij[0]
                self.r[0] = self.r[0]-100 + ij[1]
                #self.template = image[int(self.r[1]):int(self.r[1]+self.r[3]), int(self.r[0]):int(self.r[0]+self.r[2])].copy()
                #import pdb;pdb.set_trace()
                try:
                    masked[int(self.r[1]):int(self.r[1]+self.r[3]), int(self.r[0]):int(self.r[0]+self.r[2])] = 0
                except:
                    print("Failure in image {}".format(file_name))
                plt.imsave("./masked/" + file_name,masked)
                
                

if __name__== "__main__":
    handler = DataHandler("./images/")
    handler.iterative_search()
