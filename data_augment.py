import cv2
import random
import scipy
import skimage as sk
from skimage import transform
from skimage import util
import numpy as np
import lbp3
"""
def data_augmentation(dictionary):
    array=dictionary[x].copy()
    flip=np.random.rand(len(array))

    for i in range(0,len(array)):
        image=array[i]
        if flip[i]>=0.5:
            image=cv2.flip(image,1)
            array[i]=image

        #cv2.imshow("e"+str(i),array[i])
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    return dictionary
"""

def random_rotation(image: scipy.ndarray):
    x=random.randint(-10,10)
    y=random.randint(-10,10)
    random_degree = random.uniform(x,y)
    return sk.transform.rotate(image, random_degree)

#in small images noise blurs the image
#def random_noise(image_array: scipy.ndarray):
    # add random noise to the image
#    return sk.util.random_noise(image_array)

def horizontal_flip(image: scipy.ndarray):
    if random.random()>0.5:
        return image[:, ::-1]
    else:
        return image


def translation(image):
    x=random.randint(-8,8)
    y=random.randint(-8,8)
    return scipy.ndimage.shift(image,(x,y))

def augment(image,number_of_copy,size):
    lbp=[]
    augm=[]

    for i in range(0,number_of_copy):
        augmented_image=image.copy()
        augmented_image=random_rotation(augmented_image)
        augmented_image=horizontal_flip(augmented_image)
        augmented_image=translation(augmented_image)
        augmented_image=cv2.resize(augmented_image,(size,size))
        lbp_img=lbp3.main(augmented_image)

        lbp_img=cv2.resize(lbp_img,(size,size))
        augm.append(augmented_image)
        lbp.append(lbp_img)
    return (augm,lbp)
