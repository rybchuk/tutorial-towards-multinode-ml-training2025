import glob
import h5py
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

class GetClimateDataset(Dataset):
    '''Dataloader class for climate datasets'''
    def __init__(self, location, train, transform, upscale_factor, noise_ratio, std, method):
        self.location = location
        self.upscale_factor = upscale_factor
        self.train = train
        self.noise_ratio = noise_ratio
        self.std = torch.Tensor(std).view(len(std),1,1)
        self.transform = transform
        self._get_files_stats()
        self.method = method

        if method == "bicubic":
            self.bicubicDown_transform = transforms.Resize(
                (int((self.img_shape_x-1)/upscale_factor), 
                 int(self.img_shape_y/upscale_factor)),
                Image.BICUBIC,
                antialias=False
            )

    def _get_files_stats(self):
        self.files_paths = list(self.location.glob('*.h5'))
        self.files_paths.sort()
        self.n_years = len(self.files_paths)
        with h5py.File(self.files_paths[0], 'r') as _f:
            print("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_year = _f['fields'].shape[0]
            self.n_in_channels = _f['fields'].shape[1]
            self.img_shape_x = _f['fields'].shape[2]
            self.img_shape_y = _f['fields'].shape[3]

        self.n_samples_total = self.n_years * self.n_samples_per_year
        self.files = [None for _ in range(self.n_years)]
        print("Number of samples per year: {}".format(self.n_samples_per_year))
        print("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(
            self.location, self.n_samples_total, self.img_shape_x, self.img_shape_y, self.n_in_channels))

    def _open_file(self, year_idx):
        _file = h5py.File(self.files_paths[year_idx], 'r')
        self.files[year_idx] = _file['fields']  

    def __len__(self):
        if self.train == True: 
            return self.n_samples_total
        return self.n_samples_total

    def __getitem__(self, global_idx):
        year_idx = global_idx // self.n_samples_per_year
        local_idx = global_idx % self.n_samples_per_year

        # Open image file if it's not already open
        if self.files[year_idx] is None:
            self._open_file(year_idx)

        # Apply transform and cut-off
        y = self.transform(self.files[year_idx][local_idx])
        y = y[:,:-1,:]
        
        # Get X based on method (no cropping)
        X = self.get_X(y)

        return X, y

    def get_X(self, y):
        if self.method == "uniform":
            X = y[:, ::self.upscale_factor, ::self.upscale_factor]
        elif self.method == "noisy_uniform":
            X = y[:, ::self.upscale_factor, ::self.upscale_factor]
            X = X + self.noise_ratio * self.std * torch.randn(X.shape)
        elif self.method == "bicubic":
            X = self.bicubicDown_transform(y)
        else:
            raise ValueError(f"Invalid method: {self.method}")

        return X