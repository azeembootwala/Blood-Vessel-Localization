
## Blood vessel localization using Deep Convolutional Neural Networks

### Motivation
The aim of this project is to train deep neural networks to identify locations of carotid arteries in urtrasound images using bounding boxes. 

### Network Architecture
For this project [**VGG**](https://arxiv.org/pdf/1409.1556.pdf) Network architecture was refactored to add more specific layers to suit localization tasks.The Neural network was trained from scratch.

### File Descriptions
THe folder `src` contains all the source codes for the project.
 
The script `CNN_vessel.py` is the main script where the neural network is designed in a class based format.It will train your model , test it and also save the model in the results folder. 

The script `us_datagenerator.py` loads the date from the disk, preprocesses it and sends it to the CNN_vessel in batches. 

**Note:- Within the scripts , you would be required to add the paths of files you want to store, like the results, or your saved model.Currently the locations are filled by dummy paths. Please note, saved models cannot be shared or uploaded due to non-disclosures**  

The script `utilities.py`, has small helper functions needeed to run the main script file `CNN_vessel.py`.

The script `scroll.py`, just scrolls the images in the results folder and runs a slide show for presentations. So nothing important there. 

The scripts `cnn_exp.py` is just another iteration of the mail file, where I implemented batch normalization and dropout. You can skip that if you want as the results were not upto the mark after applying batch norm.


## Results
 








   
