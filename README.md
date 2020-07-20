# Art.ai
NeuralStyleTransfer

<br/>

## Description
This project is a Deep Learning based Web application, that helps demonstrate and simulate the concept of [Neural Style Transfer](https://en.wikipedia.org/wiki/Neural_Style_Transfer). Neural style transfer is an optimization technique used to take two images — a content image and a style reference image (such as an artwork by a famous painter) — and blend them together so the output image looks like the content image, but “painted” in the style of the style reference image. The project is based on [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) paper using TensorFlow & Keras.

<br/>

## Getting Started
1. Clone [this](https://github.com/Rohit-Jain-2801/Art.ai) repo.
2. Make sure all necessary [dependencies](https://github.com/Rohit-Jain-2801/Art.ai/blob/master/requirements.txt) are installed.
3. Dive into the project folder & run `python run.py` in your terminal. The flask server will start at port `5000`, by defualt.
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

<br/>

## References
* [TensorFlow - Neural Style Transfer](https://www.tensorflow.org/tutorials/generative/style_transfer?hl=en)