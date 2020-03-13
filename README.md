# Deep Autoencoder Data Compression

Autoencoder is a type of neural-network architecture, where it takes an input and tries to reproduce it while extracting the useful and salient features of the data. As naive as it may sound. the applications of autoencoders are astonishing. Image denoising, and data compression are two most prominent uses of autoencoders in present practices.

This repository is a study of various underweight autoencoder models judged on a task of compressing data, from 4-dimensions to 3-dimensions.

Before I start with the characteristics and analysis of each model, I'd like to share that training the models was extremely educational and insightful. The hyperparameter tuning was very challenging but finding the right set of parameters was worth it all. I hope you'd like my work.

### Model 1 (Dense Network)

* `Epochs`: 500
* `Batch_size`: 64
* `Activation`: ReLU
* `Learning_rate`: 0.001
* `Batch_normalization`: Yes (with momentum = 0.9)
* `Architecture`: (4, 200, 100, 50, 3, 50, 100, 200, 4)
* `Initializer`: Glorot Uniform for kernels and Glorot Normal for biases.

This was my first attempt at training an autoencoder for data compression on the given data.

![Training Loss Curve](/Graphs_and_Records/Summary_Total_Loss_s_1.svg)
