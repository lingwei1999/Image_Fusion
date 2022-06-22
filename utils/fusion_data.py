import cv2
import torch
from kornia.utils import image_to_tensor
from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class FusionData(Dataset):
    """
    Loading fusion data from hard disk.
    """

    def __init__(self, folder: Path, mode='train', transforms=lambda x: x):
        super(FusionData, self).__init__()

        assert mode in ['eval', 'train'], 'mode should be "eval" or "train"'
        names = (folder / f'{mode}.txt').read_text().splitlines()
        self.samples = [{
            'name': name,
            'img': folder / f'{name}', ## change file extension here
        } for name in names]
        
        self.transforms = transforms


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        im = self.imread(sample['img'])

        im = self.transforms(im)
        sample = {'name': sample['name'], 'im': im}

        return sample

    @staticmethod
    def imread(path: Path) -> Tensor:
        img_n = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img_t = image_to_tensor(img_n / 255.).float()
        return img_t


if __name__ == '__main__':
    fd = FusionData(folder=Path('../datasets/FLIR_ADAS_v2'))
    s = fd[0]
    print(s)
