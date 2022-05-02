import argparse
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch


from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


MODEL_ARGS = {
    'image_size': 64,
    'num_channels': 128,
    'num_res_blocks': 3,
    'num_heads': 4,
    'num_heads_upsample': -1,
    'attention_resolutions': '16,8',
    'dropout': 0.0,
    'learn_sigma': True,
    'sigma_small': False,
    'class_cond': False,
    'diffusion_steps': 4000,
    'noise_schedule': 'cosine',
    'timestep_respacing': '',
    'use_kl': False,
    'predict_xstart': False,
    'rescale_timesteps': True,
    'rescale_learned_sigmas': True,
    'use_checkpoint': False,
    'use_scale_shift_norm': True
}


def img2np(img):
    img_np = img.detach().cpu().numpy()
    img_np = ((img_np + 1) * 127.5).clip(0, 255).astype('uint8')
    img_np = img_np[0].transpose([1, 2, 0])
    return img_np

def img2torch(img_np):
    img = img_np.transpose([2, 0, 1])[None]
    img = torch.tensor(img, device='cuda')
    img = img / 127.5 - 1
    return img.float()


def main(args):
    if args.input_image is not None:
        input_img = cv2.imread(args.input_image)[:, :, ::-1]
        input_img = cv2.resize(
            input_img,
            (MODEL_ARGS['image_size'], MODEL_ARGS['image_size']),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        input_img = np.zeros(
            (MODEL_ARGS['image_size'], MODEL_ARGS['image_size'], 3),
            dtype='uint8'
        )

    input_img_mask = torch.tensor(input_img != 0, dtype=torch.float, device='cuda')
    input_img_mask = input_img_mask.max(2).values[None, None]
    input_img = img2torch(input_img)
    input_img = input_img.tile([args.batch_size, 1, 1, 1])

    model, diffusion = create_model_and_diffusion(**MODEL_ARGS)
    model.load_state_dict(
        torch.load(args.model_path)
    )
    model.cuda()
    model.eval()

    img = torch.randn(
        [args.batch_size, 3, MODEL_ARGS['image_size'], MODEL_ARGS['image_size']],
        device='cuda'
    )
    history = [img2np(img)]

    indices = range(MODEL_ARGS['diffusion_steps'] - 1, -1, -1)

    for i in tqdm(indices):
        t = torch.tensor([i] * args.batch_size, device='cuda')
        with torch.no_grad():
            out = diffusion.p_sample_image_completion(
                model,
                img,
                t,
                input_img,
                input_img_mask,
            )
            img = out["sample"]

        if i % (len(indices) // 22) == 0:
            history.append(img2np(img))

    if args.batch_size > 1:
        history = [
            img2np(img[i: i + 1])
            for i in range(img.shape[0])
        ]

    np.save('output_images.npy', np.array(history))

    h = 4
    w = len(history) // h + min(len(history) % h, 1)

    fig, axes = plt.subplots(h, w, figsize=(w * 3, h * 3))
    for i in range(len(history)):
        axes[i // w, i % w].imshow(history[i])
        axes[i // w, i % w].axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_image", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    
    args = parser.parse_args()

    main(args)