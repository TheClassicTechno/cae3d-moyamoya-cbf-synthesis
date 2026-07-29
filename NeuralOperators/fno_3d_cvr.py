"""
FNO 3D for CVR: model, data loading. Checkpoint uses in_ch=4, out_ch=1, width=64.
"""
import os
import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

TARGET_SIZE = (128, 128, 64)
DATA_DIR = _REPO_ROOT


def load_volume(path, target_size=TARGET_SIZE):
    import nibabel as nib
    from scipy.ndimage import zoom
    nii = nib.load(path)
    vol = np.asarray(nii.dataobj).astype(np.float32).squeeze()
    assert vol.ndim == 3
    factors = [target_size[i] / vol.shape[i] for i in range(3)]
    vol = zoom(vol, factors, order=1)
    vmin, vmax = vol.min(), vol.max()
    vol = (vol - vmin) / (vmax - vmin + 1e-8)
    return vol.astype(np.float32)


def pre_to_post(pre_path):
    base = os.path.basename(pre_path).replace("pre_", "post_")
    dirname = os.path.dirname(pre_path).replace("/pre", "/post")
    return os.path.join(dirname or "post", base)


def add_position_channels(vol):
    """(H,W,D) -> (4, H, W, D): image, then normalized x,y,z in [0,1]."""
    H, W, D = vol.shape
    x = np.linspace(0, 1, H, dtype=np.float32)
    y = np.linspace(0, 1, W, dtype=np.float32)
    z = np.linspace(0, 1, D, dtype=np.float32)
    xg, yg, zg = np.meshgrid(x, y, z, indexing="ij")
    return np.stack([vol, xg, yg, zg], axis=0)


class SimpleFNO3D(nn.Module):
    def __init__(self, in_ch=4, out_ch=1, modes=12, width=64):
        super().__init__()
        self.fc0 = nn.Linear(in_ch, width)
        self.conv = nn.Conv3d(width, width, 3, padding=1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_ch)
        self.width = width

    def forward(self, x):
        B, C, H, W, D = x.shape
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = torch.relu(x)
        x = self.conv(x)
        x = torch.relu(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class VolumePairs(torch.utils.data.Dataset):
    def __init__(self, pre_paths, target_size=TARGET_SIZE):
        self.pre_paths = [p for p in pre_paths if os.path.exists(pre_to_post(p))]
        self.target_size = target_size

    def __len__(self):
        return len(self.pre_paths)

    def __getitem__(self, i):
        pre = load_volume(self.pre_paths[i], self.target_size)
        post = load_volume(pre_to_post(self.pre_paths[i]), self.target_size)
        pre_4ch = add_position_channels(pre)
        return torch.from_numpy(pre_4ch).float(), torch.from_numpy(post[np.newaxis]).float()


class VolumePairsFromPairs(torch.utils.data.Dataset):
    """Dataset from explicit (pre_path, post_path) pairs (e.g. 2020 split JSON). Same interface as VolumePairs."""
    def __init__(self, pairs, target_size=TARGET_SIZE):
        self.pairs = list(pairs)
        self.target_size = target_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        pre_path, post_path = self.pairs[i]
        pre = load_volume(pre_path, self.target_size)
        post = load_volume(post_path, self.target_size)
        pre_4ch = add_position_channels(pre)
        return torch.from_numpy(pre_4ch).float(), torch.from_numpy(post[np.newaxis]).float()


# Week7: 91x109x91 + brain mask, padded to 96x112x96 for FNO
WEEK7_ORIGINAL_SHAPE = (91, 109, 91)
WEEK7_PAD_SHAPE = (96, 112, 96)


def _pad_vol_to_96_112_96(vol):
    """vol (H,W,D) e.g. (91,109,91) -> (96,112,96) with zero padding."""
    h, w, d = vol.shape
    out = np.zeros(WEEK7_PAD_SHAPE, dtype=vol.dtype)
    out[:h, :w, :d] = vol
    return out


class Week7VolumePairsFNO(torch.utils.data.Dataset):
    """Week7 pairs: load via week7_preprocess (91x109x91, brain mask), pad to 96x112x96, add position channels."""
    def __init__(self, pairs):
        self.pairs = list(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        import sys
        sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
        from week7_preprocess import load_volume as w7_load
        from week7_preprocess import TARGET_SHAPE
        pre_path, post_path = self.pairs[i]
        pre = w7_load(pre_path, target_shape=TARGET_SHAPE, apply_mask=True)
        post = w7_load(post_path, target_shape=TARGET_SHAPE, apply_mask=True)
        pre = _pad_vol_to_96_112_96(pre)
        post = _pad_vol_to_96_112_96(post)
        pre_4ch = add_position_channels(pre)
        return torch.from_numpy(pre_4ch).float(), torch.from_numpy(post[np.newaxis]).float()
