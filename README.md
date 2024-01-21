![Static Badge](https://img.shields.io/badge/python-3.7-blue)
![Static Badge](https://img.shields.io/badge/tensorflow-2.4-FF6F00)
![Static Badge](https://img.shields.io/badge/librosa-0.9.2-4D02A2)
![Static Badge](https://img.shields.io/badge/numpy-1.19.5-013243)
![Static Badge](https://img.shields.io/badge/pandas-1.3.5-150458)

# Anomaly Detection with Variational Autoencoder(VAE)
 VAE is similar to Autoencoder, but it differs in that it encodes input data into a probabilistic latent space, which improves the generation capability for unknown data. Simply put, while the AE’s learning task is to learn a function that will transform data into a latent vector that a decoder can easily reproduce, the VAE’s learning task is to learn a function that will generate parameters of distributions from which a latent vector that a decoder can easily reproduce can be sampled.
 <br>
 The goal of training in anomaly detection is to enable the VAE to learn the features of normal data and reconstruct them effectively. This will ensure that abnormal data cannot be reconstructed well, and thus anomalies can be detected.

# Overview
This jupyter notebook explains the results of anomaly detection for sound data with Variational Autoencoder.

>[!Note]
>You should set the `use_batch_norm` argument to `True`. Because `Nan` will appear during the training process.