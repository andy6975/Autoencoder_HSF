# Deep Autoencoder Data Compression

Autoencoder is a type of neural-network architecture, where it takes an input and tries to reproduce it while extracting the useful and salient features of the data.

This repository is a study of various underweight autoencoder models judged on a task of compressing data, from 4-dimensions to 3-dimensions.

### Model 1 (Dense Network)

* Epochs: 500
* Batch_size: 64
* Activation: ReLU
* Learning_rate: 0.001
* Batch_normalization: Yes (with momentum = 0.9)
* Architecture: (4, 200, 100, 50, 3, 50, 100, 200, 4)
* Initializer: Glorot Uniform for kernels and Glorot Normal for biases.

