# Image Reconstruction using Diffusion Model

This is the code for my [article](https://medium.com/@ruslanbaynazarov/diffusion-models-can-be-used-for-image-reconstruction-out-of-the-box-62d7fda78fb1).

Before running the code, you should install `improved-diffusion` package. Check the readme of the `improved-diffusion` folder.

Also models `imagenet64_uncond_100M_1500K.pt` and `upsample_cond_500K.pt` need to be downloaded. Links are in the `improved-diffusion` folder.

To generate 16 samples of doge, run:

```
python generate.py --model_path imagenet64_uncond_100M_1500K.pt --input_image doge.png --batch_size 16
```

To upsample these samples from 64 to 256 resolution, run:

```
python upsample.py --upsample_model_path upsample_cond_500K.pt --lowres_images generated_doge.png.npy --input_image doge.png
```
