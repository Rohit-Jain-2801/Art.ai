# Importing libraries
import tensorflow as tf
import tensorflow_hub as tf_hub

# running the model
def stylize(content_img_tensor, style_img_tensor):
    # Load image stylization module
    hub_module = tf_hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    # Stylize image
    outputs = hub_module(tf.constant(content_img_tensor), tf.constant(style_img_tensor))[0]

    return outputs.numpy()