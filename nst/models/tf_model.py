# Importing libraries
import time
import numpy as np
import tensorflow as tf

# Works for GPU-only
# --------------------------------------------------------------------------------------------------------------------
# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(1)
# print(tf.config.threading.get_inter_op_parallelism_threads())     # 0 means the system picks an appropriate number.
# print(tf.config.threading.get_intra_op_parallelism_threads())     # 0 means the system picks an appropriate number.

# tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
# print(tf.config.experimental.get_memory_growth(tf.config.list_physical_devices('GPU')[0]))
# --------------------------------------------------------------------------------------------------------------------

# Content layer we are interested in
CONTENT_LAYERS = ['block5_conv2'] 

# Style layer we are interested in
STYLE_LAYERS = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1', 
    'block4_conv1', 
    'block5_conv1'
]

NUM_CONTENT_LAYERS = len(CONTENT_LAYERS)
NUM_STYLE_LAYERS = len(STYLE_LAYERS)


def load_img(img_path):
    '''
    Load image array from the path
    '''
    img = tf.keras.preprocessing.image.load_img(path=img_path, grayscale=False, color_mode='rgb', target_size=None, interpolation='nearest')
    arr = tf.keras.preprocessing.image.img_to_array(img=img, data_format=None, dtype=None)
    return arr


def preprocess_img(img):
    '''
    Takes image array as input & performs necessary preprocessing for vgg19 model
    '''
    return tf.keras.applications.vgg19.preprocess_input(x=img, data_format=None)


def deprocess_img(img):
    '''
    Takes image array as input & reverts all the preprocessing
    '''
    # making a copy of image array
    image = img.copy()

    # checking if image is 4D - converting it to 3D
    if len(image.shape) == 4:
        image = np.squeeze(a=image, axis=0)

    # perform the inverse of the preprocessing step
    image[:, :, 0] += 103.939
    image[:, :, 1] += 116.779
    image[:, :, 2] += 123.68
    image = image[:, :, ::-1]

    # optimized image may take its values anywhere between −∞ and ∞
    # hence, clipping with range [0, 255]
    image = np.clip(a=image, a_min=0, a_max=255, out=None, where=True, casting='same_kind', order='K', subok=True)
    return image.astype('uint8')


def get_model():
    '''
    Loads vgg19 model and returns a keras-model with combined style & content outputs resp.
    '''
    # Load our model. We load pretrained VGG, trained on imagenet data
    # vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', pooling=None, classes=1000, classifier_activation='softmax')
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights=None, pooling=None, classes=1000, classifier_activation='softmax')
    # vgg.load_weights('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
    vgg.load_weights('./nst/models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
    vgg.trainable = False

    # Get output layers corresponding to style and content layers 
    style_outputs = [vgg.get_layer(name).output for name in STYLE_LAYERS]
    content_outputs = [vgg.get_layer(name).output for name in CONTENT_LAYERS]
    model_outputs = style_outputs + content_outputs

    # Build model 
    return tf.keras.models.Model(inputs=vgg.input, outputs=model_outputs, name='NeuralStyleTransfer')


def get_content_loss(target, gen_content):
    '''
    Takes target_content & generated_content and outputs content_loss
    '''
    return tf.math.reduce_mean(input_tensor=tf.square(x=tf.math.subtract(x=target, y=gen_content)), axis=None, keepdims=False)


def gram_matrix(input_tensor):
    '''
    Returns gram matrix of input tensor
    '''
    # We make the image channels first 
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(tensor=input_tensor, shape=[-1, channels])

    # the number of feature maps
    n = tf.shape(input=a)[0]

    # calculating gram
    gram = tf.linalg.matmul(a=a, b=a, transpose_a=True, transpose_b=False, adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False)

    return gram / tf.cast(x=n, dtype=tf.float32)


def get_style_loss(gram_target, gen_style):
    '''
    Takes gram_target & generated_style and outputs style_loss
    '''
    # getting gram matrix of generated_style
    gram_style = gram_matrix(input_tensor=gen_style)
    
    # / (4. * (channels ** 2) * (width * height) ** 2)
    return tf.math.reduce_mean(input_tensor=tf.square(x=tf.math.subtract(x=gram_target, y=gram_style)), axis=None, keepdims=False)


def get_feature_representations(model, content_img, style_img):
    """
    Helper function to compute our content and style feature representations.
    Takes model, content_image & style_image as inputs
    """
    # Load our images in 
    content_image = preprocess_img(img=content_img)
    style_image = preprocess_img(img=style_img)
    
    # batch compute content and style features
    style_outputs = model(style_image)
    content_outputs = model(content_image)
    
    # Get the style and content feature representations from our model  
    style_features = [style_layer[0] for style_layer in style_outputs[:NUM_STYLE_LAYERS]]
    content_features = [content_layer[0] for content_layer in content_outputs[NUM_STYLE_LAYERS:]]

    return style_features, content_features


def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    """
    This function will compute the total loss.
    
    Arguments:
        model: The model that will give us access to the intermediate layers
        loss_weights: The weights of each contribution of each loss function (style weight & content weight)
        init_image: Our initial base image. This image is what we are updating with our optimization process. We apply the gradients wrt the loss we are calculating to this image.
        gram_style_features: Precomputed gram matrices corresponding to the defined style layers of interest.
        content_features: Precomputed outputs from defined content layers of interest.
        
    Returns:
        returns the total loss, style loss, content loss
    """
    style_weight, content_weight = loss_weights
    
    # Feed our init image through our model. This will give us the content and style representations at our desired layers.
    model_outputs = model(init_image)
    
    style_output_features = model_outputs[:NUM_STYLE_LAYERS]
    content_output_features = model_outputs[NUM_STYLE_LAYERS:]
    
    style_score = 0
    content_score = 0

    # Accumulate style losses from all layers
    # Here, we equally weight each contribution of each loss layer
    weight_per_style_layer = 1.0 / float(NUM_STYLE_LAYERS)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(target_style, comb_style[0])
        
    # Accumulate content losses from all layers 
    weight_per_content_layer = 1.0 / float(NUM_CONTENT_LAYERS)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer* get_content_loss(target_content, comb_content[0])
    
    style_score *= style_weight
    content_score *= content_weight

    # Get total loss
    loss = style_score + content_score

    return loss, style_score, content_score


def compute_grads(cfg):
    '''
    Computes gradients
    '''
    with tf.GradientTape() as tape: 
        all_loss = compute_loss(**cfg)

    # Compute gradients wrt input image
    total_loss = all_loss[0]

    return tape.gradient(total_loss, cfg['init_image']), all_loss


# @tf.function()
def run_style_transfer(content_img, style_img, num_iterations=1000, content_weight=1e3, style_weight=1e-2, learning_rate=5):
    '''
    Performs Neural Style Transfer (NST)
    '''
    # get the model
    model = get_model()
    
    # Get the style and content feature representations (from our specified intermediate layers) 
    style_features, content_features = get_feature_representations(model=model, content_img=content_img, style_img=style_img)
    gram_style_features = [gram_matrix(input_tensor=style_feature) for style_feature in style_features]
    
    # Set initial image (basically that results in the generated image)
    init_image = preprocess_img(img=content_img)
    init_image = tf.Variable(initial_value=init_image, dtype=tf.float32)

    # Create our optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.99, epsilon=1e-1)
    
    # Store our best result
    best_loss, best_img = float('inf'), None
    
    # Create a nice config
    config = {
        'model': model,
        'loss_weights': (style_weight, content_weight),
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }
    
    # for clipping purpose
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means
    
    
    for i in range(num_iterations):
        s = time.time()

        # setting update flag to False
        update = False

        # computing gradients
        grads, all_loss = compute_grads(cfg=config)

        # demystifying loss
        loss, style_score, content_score = all_loss

        # applying gradients using optimizer
        opt.apply_gradients(grads_and_vars=[(grads, init_image)], experimental_aggregate_gradients=True)

        # clipping image
        clipped = tf.clip_by_value(t=init_image, clip_value_min=min_vals, clip_value_max=max_vals)

        # updating image
        init_image.assign(clipped)
        
        # comparing loss
        if loss < best_loss:
            # setting update flag to True
            update = True

            # Update best loss and best image from total loss.
            best_loss = loss
            best_img = deprocess_img(img=init_image.numpy())

        e = time.time()
        print('Epoch: {} \tLoss: {} \t\tTime: {:.4f}'.format((i+1), loss.numpy(), (e-s)))
        yield (i+1), update, best_img
    
    # update = False
    # return (i+1), update, best_img