# Face-Alignment-System
Data files are located within the release section.

Convolutional Neural Network model that becomes trained to detect facial features from grayscale face images by placing coordinates(x,y) where the key facial features should be. 

Data is loaded from training, validation and test data(without labels) which is then used to train the model.

CNN Architecture has 2 convolution layers (with maxpooling and relu activation) followed by a flattened dense layer with a dropout of 50%. 

Models loss function is MSE, optimized with ADAM.

The model is trained over 15 epochs each with a batch size of 12. 

To improve robustness, extra gaussian noise is added to increase the training data set. 

Performance has been evaluated using mean eucledian pixel error on validata data. 
results are visualised through error histograms and cumulative error distribution plots. 

The model is also stress-tested against higher noise levels than it was trained on, to probe how well it generalises under degraded conditions.
