# dlsisr

Initial implementation and testbed for SRGAN architecture changes to increase resolution of AT&T dataset images.

The network models, including VGG19 feature extractor, generator, and discriminator are in models.py.

The training, testing, and validation are handled by solver.py.

We define a custom PyTorch dataset for the AT&T images in datasets.py.

The configuration of our runs, including the pre-training step, selection of VGG19 feature layer, and other hyperparameters, are handled in main.py.

To run, edit the configuration options passed to each `main()` function in the `__main__` block. A full-training step requires a pre-training step with the pixel-wise MSE content loss followed by training with the VGG feature MSE content loss. We implement this by calling main twice, by first setting 
```
config['content_loss']='mse'
main(config)
```
then to train with VGG feature loss
```
config['content_loss']='VGG'
main(config)
```

Other configuration options are relatively straightforward. To set reconstruction loss hyperparameter use
```
config['lambda_rec'] = your_value_here
```
which defaults to 10.

REFERENCES
1. StarGAN - Official PyTorch Implementation https://github.com/yunjey/stargan
2. SRGAN-PyTorch https://github.com/Lornatang/SRGAN-PyTorch
3. PyTorch-GAN https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/srgan/srgan.py 
