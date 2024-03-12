from skimage import io, color
from skimage.transform import rescale , resize  , downscale_local_mean 
from matplotlib import pyplot as plt
import  cv2

img = cv2.imread("rgb.png",0)         
template =cv2.imread("rgb obj.png",0)    


#finding accurate object in key

h,w = template.shape[::]    
res = cv2.matchTemplate(img, template , cv2.TM_SQDIFF)        

min_val , max_val , min_loc , max_loc = cv2.minMaxLoc(res)        
top_left=min_loc
bottom_right = (top_left[0]+w , top_left[1]+h)      
cv2.rectangle(img, top_left, bottom_right,0 , 1)   
cv2.imshow("matched img" , img)
cv2.waitKey()
cv2.destroyAllWindows()


#finding object from picture with 50 percent match

import numpy as np
res = cv2.matchTemplate(img, template , cv2.TM_CCOEFF_NORMED)
threshold = 0.5  
loc=np.where(res>=threshold)       
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0]+w,pt[1]+h), (0,0,255) , 1)
    print(pt)
 
    
cv2.imshow("matched img" , img)
cv2.waitKey()
cv2.destroyAllWindows()