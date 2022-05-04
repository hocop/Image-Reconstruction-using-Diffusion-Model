import argparse
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch


from improved_diffusion.script_util import (
    sr_create_model_and_diffusion,
)

UPS_MODEL_ARGS = {
    'large_size': 256,
    'small_size': 64,
    'num_channels': 192,
    'num_res_blocks': 2,
    'num_heads': 4,
    'num_heads_upsample': -1,
    'attention_resolutions': '16,8',
    'dropout': 0.0,
    'learn_sigma': True,
    'class_cond': True,
    'diffusion_steps': 4000,
    'noise_schedule': 'linear',
    'timestep_respacing': '',
    'use_kl': False,
    'predict_xstart': False,
    'rescale_timesteps': False,
    'rescale_learned_sigmas': False,
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
    # Load image to complete
    if args.input_image is not None:
        input_img = cv2.imread(args.input_image)[:, :, ::-1]
        input_img = cv2.resize(
            input_img,
            (UPS_MODEL_ARGS['large_size'], UPS_MODEL_ARGS['large_size']),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        input_img = np.zeros(
            (UPS_MODEL_ARGS['large_size'], UPS_MODEL_ARGS['large_size'], 3),
            dtype='uint8'
        )

    input_img_mask = torch.tensor(input_img != 0, dtype=torch.float, device='cuda')
    input_img_mask = input_img_mask.max(2).values[None, None]
    input_img = img2torch(input_img)

    # Load low-res images
    if args.lowres_images.endswith('.npy'):
        imgs_low_np = np.load(args.lowres_images)
    else:
        imgs_low_np = cv2.imread(args.lowres_images)[:, :, ::-1]
        imgs_low_np = cv2.resize(
            imgs_low_np,
            (UPS_MODEL_ARGS['large_size'], UPS_MODEL_ARGS['large_size']),
        )
        imgs_low_np = [imgs_low_np]

    # Load model
    ups_model, ups_diffusion = sr_create_model_and_diffusion(**UPS_MODEL_ARGS)
    ups_model.load_state_dict(
        torch.load(args.upsample_model_path)
    )
    ups_model.cuda()
    ups_model.eval()

    results = []

    for img_low in imgs_low_np:
        img_low = img2torch(img_low)
        batch_size = 1

        img = torch.randn(
            [batch_size, 3, UPS_MODEL_ARGS['large_size'], UPS_MODEL_ARGS['large_size']],
            device='cuda'
        )
        history = [img2np(img)]

        indices = range(UPS_MODEL_ARGS['diffusion_steps'] - 1, -1, -1)

        y = torch.tensor([args.class_idx] * batch_size, device='cuda')

        for i in tqdm(indices):
            t = torch.tensor([i] * batch_size, device='cuda')
            with torch.no_grad():
                out = ups_diffusion.p_sample_image_completion(
                    ups_model,
                    img,
                    t,
                    input_img,
                    input_img_mask,
                    model_kwargs={
                        'low_res': img_low,
                        'y': y,
                    }
                )
                img = out["sample"]

            if i % (len(indices) // 22) == 0:
                history.append(img2np(img))
        
        results.append(history[-1])

    if len(imgs_low_np) > 1:
        history = results

    np.save(f'upsampled_{args.input_image}.npy', np.array(history))

    h = 4
    w = len(history) // h + min(len(history) % h, 1)

    fig, axes = plt.subplots(h, w, figsize=(w * 4, h * 4))
    for i in range(len(history)):
        axes[i // w, i % w].imshow(history[i])
        axes[i // w, i % w].axis('off')
    plt.tight_layout()
    # plt.show()
    fig.savefig(f'upsampled_{args.input_image}.png', dpi=fig.dpi)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--lowres_images", type=str, default=None)
    parser.add_argument("--input_image", type=str, default=None)
    parser.add_argument("--upsample_model_path", type=str)
    parser.add_argument("--class_idx", type=int, default=959)
    
    args = parser.parse_args()

    main(args)
