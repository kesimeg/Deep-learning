import cv2
import random
import scipy
import skimage as sk
from skimage import transform
from skimage import util
import numpy as np
import lbp3


def random_rotation(image: scipy.ndarray):
    x=random.randint(-15,15)
    y=random.randint(-15,15)
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


def translation(image,xt,yt):
    x=random.randint(-xt,xt)
    y=random.randint(-yt,yt)
    #print(x,y)
    return scipy.ndimage.shift(image,(x,y))

def random_noise(image):
    shape_of=np.array(image).shape
    new_image=image.copy()
    iter=int(shape_of[0]*shape_of[1]/35)
    for i in range(0,iter):
        new_image[random.randint(0,shape_of[0]-1),random.randint(0,shape_of[1]-1)]=random.random()
    return new_image

def random_color_shift(image):
    mean_val=np.mean(image)
    if mean_val>130:
        ran=np.random.randint(size=1,low=255,high=320)
    else:
        ran=np.random.randint(size=1,low=130,high=250)
    new_image=image.copy()

    new_image=np.array(new_image)
    new_image=new_image/ran
    #print(new_image)
    return new_image

def augment(image,number_of_copy,size):
    lbp=[]
    augm=[]

    for i in range(0,number_of_copy):
        augmented_image=image.copy()
        augmented_image=random_color_shift(augmented_image)
        augmented_image=random_rotation(augmented_image)
        augmented_image=horizontal_flip(augmented_image)
        augmented_image=translation(augmented_image,10,10)
        augmented_image=random_noise(augmented_image)
        augmented_image=cv2.resize(augmented_image,(size,size))
        lbp_img=lbp3.main(augmented_image)

        lbp_img=cv2.resize(lbp_img,(size,size))
        augm.append(augmented_image)
        lbp.append(lbp_img)
    return (augm,lbp)
