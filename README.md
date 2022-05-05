# dlsisr

Initial implementation and testbed for SRGAN architecture changes to increase resolution of AT&T dataset images.

The network models, including VGG19 feature extractor, generator, and discriminator are in models.py.

The training, testing, and validation are handled by solver.py.

We define a custom PyTorch dataset for the AT&T images in datasets.py.

## Configuring and running the code

Simply clone and run
```
python main.py
```
This code is meant to be deployed to the SCC. Training with the celebA dataset takes over 20 hours using the batch job script `main.sh`.

Since we ran by submitting batch jobs, configuration options including the pre-training step, selection of VGG19 feature layer, and other hyperparameters, are exposed only in main.py.

To run, edit the configuration options passed to each `main()` function in the `__main__` block in `main.py`. A full run involves a pre-training step with the pixel-wise MSE content loss followed by training with the VGG feature MSE content loss. We implement this by calling main twice, by first setting 
```
config['content_loss']='mse'
main(config)
```
then resume training from the last saved model and train with VGG feature loss
```
config['content_loss']='VGG'
main(config)
```

Other configuration options are relatively straightforward. To set reconstruction loss hyperparameter use
```
config['lambda_rec'] = your_value_here
```
which defaults to 10. To disable set to 0.

## Access to celebA dataset
Due in part to intermittent access to celebA via Google Drive API, we did not use the PyTorch CelebA dataset loader API. If this code is deployed somewhere else without read access to the celebA dataset at `/projectnb/dl523/materials/datasets/` on the SCC then either edit the celebA dataset loader [here](https://github.com/erhancanozcan/dlsisr/blob/7eea326eab68bea5b8b5f5816e0058b2bc7e0d8b/common/celebA.py#L58) to point to your celebA dataset.

## References
1. StarGAN - Official PyTorch Implementation https://github.com/yunjey/stargan
2. SRGAN-PyTorch https://github.com/Lornatang/SRGAN-PyTorch
3. PyTorch-GAN https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/srgan/srgan.py 
