# Color texture

Use CNN to learn how to color a picture based on its textures.

## CNN model

The structure of the CNN model is as following

![CNN-Model](./doc/CNN.png)

The input is the small patch of local image texture.
The gray channel of a picture is considered as the texture.

![Picture-compare](./doc/compare.png)

Assume we have some pictures,
the aim is to teach the CNN to learn how to color the gray-scaled textures.

![Color-results](./doc/results.png)

## Contents

The project contains the folders:

- assets: The pictures being converted with each other;
- converted: The converted pictures;
- parameters: The trained parameters of the pictures in assets.

The project contains the scripts:

- images.py: The python script to load image into Image class;
- main.py: The main python script of training the model, it also convert the pictures;
- batch.sh: The shell script of running several main.py.

The parameters of the CNN:

- The parameters are specifically to the picture;
- After the model is trained, the parameters will be saved in the parameters folder.

