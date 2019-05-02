import numpy as np
import cv2
import scipy.ndimage
import os
import data_augment

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #face detector
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


#img=cv.resize(img,(1000,1000))



#data augmentation flipping,translation,noise

def align_face(eyes,image): #aling face
    if(len(eyes)>1):
        eye1=eyes[0]
        eye2=eyes[1]
        eye1_center=[eye1[0]+eye1[2]/2,eye1[1]+eye1[3]/2]
        eye2_center=[eye2[0]+eye2[2]/2,eye2[1]+eye2[3]/2]

        dY = eye1_center[1] - eye2_center[1]
        dX = eye1_center[0] - eye2_center[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180
         #rotated=scipy.ndimage.interpolation.rotate(image,angle)

        if abs(angle)<60:
            rotated=scipy.ndimage.interpolation.rotate(image,angle)
        else:
            rotated=image
            #print(angle)

        return rotated
    else:
        return image

def face_recog(img):
    #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #A good resource:
    #https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    for (x,y,w,h) in faces:
        #cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = img[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cropped_img=img[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        return align_face(eyes,cropped_img)



path_name="./Data"
class_files=os.listdir(path_name)
class_files.sort()
print(class_files)
number_of_copies=50
image_size=72


for element in class_files:
    image_path=path_name+"/"+element
    files=os.listdir(image_path)
    for image_name in files:
        full_image_path=image_path+"/"+image_name
        image=cv2.imread(full_image_path,0)
        #print(full_image_path)
        image=face_recog(image)
        aug=data_augment.augment(image,number_of_copies,72)
        for i in range(0,number_of_copies):
                #cv2.imwrite("./ege3.tiff",aug[0][3]*255)
                name=image_name.split(".tiff")
                cv2.imwrite("./augment_data_50_copy/processed/"+element+"/"+name[0]+"_"+str(i)+".tiff",aug[0][i]*255) #has to be converted to 0-255
                cv2.imwrite("./augment_data_50_copy/lbp/"+element+"/"+name[0]+"_"+str(i)+".tiff",aug[1][i]*255)
