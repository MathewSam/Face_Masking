#%%
import numpy as np 
import matplotlib.pyplot as plt
from skimage.feature import match_template 
import cv2
#%%
image = cv2.imread("test.jpg")
r = cv2.selectROI(image)
cv2.waitKey(0)
cv2.destroyAllWindows()
template = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])][:,:,::-1]

#%%
image2 = plt.imread("test.jpg")
search_space = image2[int(r[1])-100:int(r[1]+r[3])+100, int(r[0])-100:int(r[0]+r[2])+100]
result = match_template(search_space, template)
#%%
ij = np.unravel_index(np.argmax(result), result.shape)
#%%
plt.imshow(search_space[ij[0]:ij[0]+r[3],ij[1]:ij[1]+r[2]])

#%%
