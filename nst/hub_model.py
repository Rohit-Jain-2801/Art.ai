# Importing libraries
import tensorflow as tf
import tensorflow_hub as tf_hub

# running the model
def stylize(content_img_tensor, style_img_tensor):
    # Load image stylization module
    hub_module = tf_hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    # Stylize image
    outputs = hub_module(tf.constant(value=content_img_tensor, dtype=tf.float32), tf.constant(value=style_img_tensor, dtype=tf.float32))[0]
    return outputs.numpy()[0]


def preprocess(img):
    img = tf.image.resize(images=img, size=(256, 256), method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False, antialias=False)
    return (img / 255.0)