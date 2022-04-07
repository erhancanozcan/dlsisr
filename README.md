# dlsisr

Initial implementation and testbed for SRGAN architecture changes to increase resolution of AT&T dataset images.

The network models, including VGG19 feature extractor, generator, and discriminator are in models.py.

The training, testing, and validation are handled by solver.py.

We define a custom PyTorch dataset for the AT&T images in datasets.py.

The configuration of our runs, including the pre-training step, selection of VGG19 feature layer, and other hyperparameters, are handled in main.py.
