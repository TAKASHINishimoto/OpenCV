import numpy as np
import cv2

def main():
    im = cv2.imread("sample.png")                         
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)          
    th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    k = np.ones((3,3),np.uint8)                        
    bg = cv2.dilate(th,k,iterations=3)                 
    trans = cv2.distanceTransform(bg,cv2.DIST_L2,3)    
    fg = cv2.threshold(trans,0.1*trans.max(),255,0)[1] 
    fg = np.uint8(fg)                                 
    unknown = cv2.subtract(bg,fg)
    marker = cv2.connectedComponents(fg)[1] 
    marker = marker + 1
    marker[unknown == 255] = 0              
    marker = cv2.watershed(im,marker)                  
    im[marker == -1] = [0,255,0]                      
    cv2.imshow("watershed",im)                         
    cv2.waitKey(0)                                     
    cv2.destroyAllWindows()                      

if __name__ == "__main__":
    main()
