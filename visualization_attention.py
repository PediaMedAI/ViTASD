import argparse
import functools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import torchshow
import types

from pathlib import Path
from tqdm import tqdm

from datasets import AutismDatasetModule
from lib.utils import unnormalize
from train import ViTASDLM


def forward(self, x, attn_maps: list):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    # Save attention maps here
    tensor_attn = attn[0, 0, :, :]
    attn_maps.append(torch.clone(tensor_attn).cpu().numpy())
    ########

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def vis_attn(mat: np.ndarray, output_path: Path, img=None, heatmap_kwargs=None, savefig_kwargs=None):
    if heatmap_kwargs is None:
        heatmap_kwargs = {}
    if savefig_kwargs is None:
        savefig_kwargs = {}
    # if not output_path.exists():
    #     output_path.mkdir(parents=True)

    fig, ax = plt.subplots(1)
    heatmap = sns.heatmap(mat, ax=ax, **heatmap_kwargs)
    if img is not None:
        ax.imshow(img, aspect=heatmap.get_aspect(), extent=heatmap.get_xlim() + heatmap.get_ylim(), zorder=1)
    fig.savefig(output_path, dpi=200, **savefig_kwargs)
    ax.clear()


def main(hparams):
    hparams['output_root'] = Path(hparams['output_root'])
    data_module = AutismDatasetModule(
        batch_size=1,
        data_root=hparams['data_root'],
        color_jitter=0,
        input_size=224,
        three_augment=False
    )
    model: ViTASDLM = ViTASDLM.load_from_checkpoint(hparams['ckpt_path'])
    model.eval()
    loader = data_module.predict_dataloader()
    dataset: ImageDataset = loader.dataset
    attn_maps = []
    for i in range(hparams['num_layers']):
        attn_layer: torch.nn.Module = model.model.backbone.blocks[i].attn
        forward_fn = functools.partial(forward, attn_maps=attn_maps)
        attn_layer.forward = types.MethodType(forward_fn, attn_layer)

    image_path = 'imgs/Non_Autistic/011.jpg'

    it = iter(loader)
    for image_idx in tqdm(range(len(dataset))):
        data_item = next(it)
        cur_path = str(dataset.filename(image_idx, absolute=False)).strip()
        if cur_path == image_path:
            _, cls, number = cur_path.split('/')
            number = number.split('.')[0]
            output_path = hparams['output_root'] / image_path.split('.')[0]
            # if not output_path.exists():
            #     output_path.mkdir(parents=True)

            img, target = data_item
            torchshow.save(img, str(output_path / f'input.jpg'))
            with torch.no_grad():
                pred = model(img)
                pred = F.softmax(pred[0], dim=0)
                print(pred)
            img = unnormalize(img[0]).permute(1, 2, 0).cpu().numpy()

            i = 1
            for mp in attn_maps:
                # Attentions between the distraction token and visual tokens
                mat = mp[0, :]
                heatmap_kwargs = dict(cmap="jet", zorder=2, cbar=False, xticklabels=False, yticklabels=False,
                                        alpha=0.5)
                savefig_kwargs = dict(bbox_inches='tight', pad_inches=0.01)
                mat = mat[1: 197].reshape([14, 14])
                vis_attn(mat, output_path / f'attn_{i}.png',
                            img, heatmap_kwargs, savefig_kwargs)
                i += 1

            attn_maps.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root', type=str, default=r"")
    parser.add_argument('--ckpt_path', type=str, default=r"")
    parser.add_argument('--data_root', type=str, default=r"")
    parser.add_argument('--num_layers', type=int, default=12)
    args = vars(parser.parse_args())
    main(args)
