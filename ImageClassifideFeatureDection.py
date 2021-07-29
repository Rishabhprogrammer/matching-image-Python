import cv2
import numpy as np
import os

path = 'ImgQuery' ### Your Query Image Folder Name .
orb = cv2.ORB_create(nfeatures=1000)
#### Import Imges
imges = []
classNames = []
myList = os.listdir(path)
#print("Total Classes Dectected", len(myList))
for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}',0)
    imges.append(imgCur)
    #print(cl)
    classNames.append(os.path.splitext(cl)[0])
#print(classNames)    

def findDes(imges):
    desList=[]
    for img in imges:
        kp,des = orb.detectAndCompute(img,None)
        desList.append(des)
    return desList

def findID(img, desList,thres=15):
    kp2,des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher()
    matchList=[]
    finalVal = -1
    try:
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])

            matchList.append(len(good))
    except:
        pass
    #print(matchList)
    if len(matchList)!=0:
        if max(matchList) > thres:
            finalVal = matchList.index(max(matchList))
    return finalVal        

desList = findDes(imges)
#print(len(desList))

cap = cv2.VideoCapture(0)

while True:

    success, img2 = cap.read()
    imgOrginal = img2.copy()
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    
    id = findID(img2,desList)
    if id != -1:
        cv2.putText(imgOrginal,classNames[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
    
    cv2.imshow('img2',imgOrginal)
    cv2.waitKey(1)
    