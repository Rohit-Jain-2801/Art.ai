# Art.ai
NeuralStyleTransfer

<br/>

## Description
This project is a Deep Learning based Web application, that helps demonstrate and simulate the concept of [Neural Style Transfer](https://en.wikipedia.org/wiki/Neural_Style_Transfer). Neural style transfer is an optimization technique used to take two images — a content image and a style reference image (such as an artwork by a famous painter) — and blend them together so the output image looks like the content image, but “painted” in the style of the style reference image. The project is based on [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) paper using TensorFlow & Keras.

<br/>

## Getting Started
1. Clone [this](https://github.com/Rohit-Jain-2801/Art.ai) repo.
2. Make sure all necessary [dependencies](https://github.com/Rohit-Jain-2801/Art.ai/blob/master/requirements.txt) are installed.
3. Dive into the project folder & run `python run.py` in your terminal. The flask server will start at port `5000`, by default.
4. Upload content & style images.
5. Configure the settings - choose between `TensorFlow-Hub` (default) or `TensorFlow Manual Training`.
6. Sit tight & monitor the progress.

<br/>

## Major Tech Stack
* [Python](https://www.python.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [Flask](https://flask.palletsprojects.com/en/1.1.x/)
* [Socket.io](https://socket.io/)
* [MaterializeCSS](https://materializecss.com/)

<br/>

## Future Scope
* Provide option for selecting curated style images.
* Improve speed of the manual training.
* Provide option for sharing images.
* Applying NST on videos.

<br/>

## References
* [TensorFlow - Neural Style Transfer](https://www.tensorflow.org/tutorials/generative/style_transfer?hl=en)
* [PyImageSearch - Neural Style Transfer with OpenCV](https://www.pyimagesearch.com/2018/08/27/neural-style-transfer-with-opencv/)
* Medium Posts-
  + [Chi-Feng Wang - A Basic Introduction to Separable Convolutions](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728#:~:text=Unlike%20spatial%20separable%20convolutions%2C%20depthwise,factored%E2%80%9D%20into%20two%20smaller%20kernels.&text=The%20depthwise%20separable%20convolution%20is,number%20of%20channels%20%E2%80%94%20as%20well.)
  + [Sahil Singla - Experiments on different loss configurations for style transfer](https://towardsdatascience.com/experiments-on-different-loss-configurations-for-style-transfer-7e3147eda55e)
  + [Sahil Singla - Practical techniques for getting style transfer to work](https://towardsdatascience.com/practical-techniques-for-getting-style-transfer-to-work-19884a0d69eb)
* StackOverflow Blogs-
  + [How can I access my localhost from my Android device?](https://stackoverflow.com/questions/4779963/how-can-i-access-my-localhost-from-my-android-device)
  + [Configure Flask dev server to be visible across the network](https://stackoverflow.com/questions/7023052/configure-flask-dev-server-to-be-visible-across-the-network)