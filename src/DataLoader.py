import h5py, torch, csv, numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from os.path import join
from PIL import Image

class CelebDataSet(Dataset):
    """
    Fast split-aligned version.
    Returns: (x2, x4, hr, lr, heatmap) with heatmap shape (1,128,128).
    """
    def __init__(
        self,
        data_path: str = './dataset/',
        state: str = 'train',
        data_augmentation: bool = False,
        heatmap_h5: str = None,
        preload_split: bool = True,
        chunk: int = 8192,          # used only for preloading split
        use_bicubic: bool = False,
    ):
        self.main_path = data_path
        self.state = state
        self.data_augmentation = data_augmentation
        self.img_path = join(self.main_path, './img_align_celeba/')
        self.eval_partition_path = join(self.main_path, 'list_eval_partition.csv')
        self.use_bicubic = use_bicubic

        interpolation_mode = InterpolationMode.BICUBIC if use_bicubic else InterpolationMode.BILINEAR

        # --- split lists ---
        train_list, val_list, test_list = [], [], []
        with open(self.eval_partition_path, 'r') as f:
            reader = csv.reader(f)
            for fname, split in reader:
                fname, split = fname.strip(), split.strip()
                if split == '0':
                    train_list.append(fname)
                elif split == '1':
                    val_list.append(fname)
                else:
                    test_list.append(fname)

        if state == 'train':
            self.image_list = sorted(train_list)
        elif state == 'val':
            self.image_list = sorted(val_list)
        else:
            self.image_list = sorted(test_list)

        # --- transforms (unchanged) ---
        if state=='train' and data_augmentation:
            self.pre_process = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop((178,178)),
                transforms.Resize((128,128)),
                transforms.RandomRotation(20, interpolation=interpolation_mode),
                transforms.ColorJitter(0.4,0.4,0.4,0.1)
            ])
        else:
            self.pre_process = transforms.Compose([
                transforms.CenterCrop((178,178)),
                transforms.Resize((128,128),
                interpolation=interpolation_mode)
            ])

        self.totensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
        self.down64 = transforms.Resize((64,64), interpolation=interpolation_mode)
        self.down32 = transforms.Resize((32,32), interpolation=interpolation_mode)
        self.down16 = transforms.Resize((16,16), interpolation=interpolation_mode)

        # --- heatmaps (split-aligned) ---
        self.heatmaps_split = None
        self._lazy_h5 = None
        self._lazy_ds = None

        if heatmap_h5:
            with h5py.File(heatmap_h5, 'r') as h5f:
                ds = h5f['heatmaps']          # shape (N_all, 128, 128), dtype e.g. float32/uint8
                H, W = int(ds.shape[1]), int(ds.shape[2])

                if 'filenames' in h5f:
                    h5_names = [n.decode() if isinstance(n, bytes) else str(n) for n in h5f['filenames'][...]]
                    name2idx = {fn: i for i, fn in enumerate(h5_names)}
                else:
                    import os
                    exts = ('.jpg', '.jpeg', '.png')
                    all_fnames = sorted([f for f in os.listdir(self.img_path) if f.lower().endswith(exts)])
                    name2idx = {fn: i for i, fn in enumerate(all_fnames)}

                split_idxs = []
                missing = []
                for fn in self.image_list:
                    gi = name2idx.get(fn, -1)
                    if gi < 0:
                        missing.append(fn)
                    split_idxs.append(gi)
                split_idxs = np.asarray(split_idxs, dtype=np.int64)

                if missing:
                    print(f"[warn] {len(missing)} filenames of split not found in HDF5, zero-filling those heatmaps.")

                if preload_split:
                    dtype = ds.dtype
                    C = 1
                    self.heatmaps_split = np.zeros((len(self.image_list), C, H, W), dtype=dtype)
                    valid_mask = split_idxs >= 0
                    valid_idxs = split_idxs[valid_mask]
                    split_pos  = np.nonzero(valid_mask)[0]

                    for off in range(0, len(valid_idxs), chunk):
                        take = valid_idxs[off:off+chunk]
                        pos  = split_pos[off:off+chunk]
                        for i, (gi, p) in enumerate(zip(take, pos)):
                            self.heatmaps_split[p, 0, :, :] = ds[gi, :, :]
                else:
                    # Lazy path: keep file handle to read on demand
                    self._lazy_h5 = h5py.File(heatmap_h5, 'r')
                    self._lazy_ds = self._lazy_h5['heatmaps']
                    self._lazy_idxs = split_idxs
                    self._lazy_shape = (1, H, W)
                    self._lazy_dtype = self._lazy_ds.dtype

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # image path → PIL
        fname = self.image_list[index]
        img = Image.open(join(self.img_path, fname)).convert('RGB')
        img = self.pre_process(img)

        # multiscale
        x4 = self.down64(img)
        x2 = self.down32(x4)
        lr = self.down16(x2)

        # to tensors
        hr_tensor = self.totensor(img)
        x4_tensor = self.totensor(x4)
        x2_tensor = self.totensor(x2)
        lr_tensor = self.totensor(lr)

        # heatmap (split-aligned)
        if self.heatmaps_split is not None:
            hm_np = self.heatmaps_split[index]             # (1,H,W), np dtype from disk
            heat = torch.as_tensor(hm_np)                  # no copy
        elif self._lazy_ds is not None:
            gi = self._lazy_idxs[index]
            if gi >= 0:
                hm = self._lazy_ds[gi, :, :]               # (H,W)
                heat = torch.from_numpy(np.expand_dims(hm, 0))
            else:
                heat = torch.zeros(1, 128, 128, dtype=torch.float32) # missing/fallback
        else:
            heat = torch.zeros(1, 128, 128, dtype=torch.float32) # no heatmaps

        return x2_tensor, x4_tensor, hr_tensor, lr_tensor, heat
