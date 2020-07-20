# Importing libraries
import tensorflow as tf
import tensorflow_hub as tf_hub


def preprocess(img):
    '''
    Takes image array and outputs a resized & normalized image as required by hub-model
    '''
    img = tf.image.resize(images=img, size=(256, 256), method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False, antialias=False)
    return (img / 255.0)


def stylize(content_img_tensor, style_img_tensor):
    '''
    Loads hub-model & applies on the images
    '''
    # Load image stylization module
    hub_module = tf_hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    # Stylize image
    outputs = hub_module(tf.constant(value=content_img_tensor, dtype=tf.float32), tf.constant(value=style_img_tensor, dtype=tf.float32))[0]
    return outputs.numpy()[0]


def run_style_transfer(content_img, style_img):
    '''
    Performs Neural Style Transfer (NST)
    '''
    content = preprocess(img=content_img)
    style = preprocess(img=style_img)

    return stylize(content_img_tensor=content, style_img_tensor=style)