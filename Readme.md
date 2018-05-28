
## Blood vessel localization using Deep Convolutional Neural Networks

### Motivation
The aim of this project is to train deep neural networks to identify locations of carotid arteries in urtrasound images using bounding boxes. 

### Network Architecture
For this project [**VGG**](https://arxiv.org/pdf/1409.1556.pdf) Network architecture was refactored to add more specific layers to suit localization tasks.The Neural network was trained from scratch.

### File Descriptions
The script `CNN_vessel.py` is the main script where the neural network is designed in a class based format.It will train your model , test it and also save the model in the results folder. 

The script us_datagenerator loads the date from the disk, preprocesses it and sends it to the CNN_vessel in batches. 

The script utilities, has  nothing but small functions needeed to run the main script file CNN_vessel.

The script scroll , just scrolls the images in the results folder and runs a slide show for presentations. So nothing important there. 

The scripts cnn_exp is just another iteration of the mail file , where I implemented batch normalization and dropout. You can skip that if you want as the results were not upto the mark after applying batch norm. 








   
