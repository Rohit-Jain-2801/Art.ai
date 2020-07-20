# Importing libraries
import io
import cv2
import base64
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def base_to_img(data):
    '''
    Converts base64 to numpy array image
    '''
    imgdata = base64.b64decode(s=bytes(data['binary'], 'utf-8'))
    image = Image.open(fp=io.BytesIO(initial_bytes=imgdata), mode='r')    
    img = cv2.cvtColor(src=np.array(object=image), code=cv2.COLOR_BGR2RGB)
    return np.expand_dims(a=img, axis=0)

def img_to_baseuri(img):
    '''
    Converts numpy array image to base64 uri
    '''
    img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = cv2.cvtColor(src=img, code=cv2.COLOR_RGB2BGR, dst=None)
    _, enc = cv2.imencode(ext='.jpeg', img=img)
    output = base64.b64encode(s=enc.tobytes())
    return u'data:image/jpeg;base64,' + str(output, 'utf-8')

def save(img):
    '''
    Saves image on the server
    '''
    plt.imsave(fname='img.jpeg', arr=img)