# GCN-NMT
neural machine translation using graph convolutional encoder

## Encoder

RNN + GCN 

RNN extract features from each word and the features are used in GCN as inputs
Then, GCN aggregate informations of neighborhood defined by adjacency matrix 

Adjacency matrix is composed of dependency tree between two nodes (words)


## Decoder

simple RNN from Bahnadau et al. (2015)

# Implementation

jupyter notebook code named "GCN+NMT.ipynb"

## Reference

1. J. Bastings, I.Titov, W.Aziz, D.Marcheggiani, K.Sima'an, "Graph Convolutional Encoders for Syntax-aware Neural Machine Translation," ICLR, 2017 [pdf](https://arxiv.org/pdf/1704.04675.pdf)

2. D. Bahdanau, K. Cho, and Y. Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate," ICLR, 2015 [pdf](https://arxiv.org/abs/1409.0473)

3. tkipf's repo : https://github.com/tkipf/pygcn
