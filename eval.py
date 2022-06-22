import argparse

import logging
from pathlib import Path

import torch
from argparse import Namespace

from kornia.losses import SSIMLoss
from torch import nn,Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.attention import Attention
from models.constructor import Constructor
from models.extractor import Extractor

from utils.fusion_data import FusionData

class Eval:

    def __init__(self, model_path: str, config: dict):
        logging.info(f'Eval')

        # device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # model parameters
        params = torch.load(model_path, map_location='cpu')

        # load extractor
        net_ext = Extractor()
        net_ext.load_state_dict(params['ext'])
        net_ext.to(device)
        net_ext.eval()
        self.net_ext = net_ext

        # load constructor
        net_con = Constructor()
        net_con.load_state_dict(params['con'])
        net_con.to(device)
        net_con.eval()
        self.net_con = net_con

        # load attention layer
        net_att = Attention()
        net_att.load_state_dict(params['att'])
        net_att.to(device)
        net_att.eval()
        self.net_att = net_att

        # loss
        self.mse = nn.MSELoss(reduction='none')
        self.ssim = SSIMLoss(window_size=11, reduction='none')
        self.mse.cuda()
        self.ssim.cuda()
        # datasets
        folder = Path(config.folder)
        resize = transforms.Resize((config.size, config.size))

        eval_dataset = FusionData(folder, mode='eval', transforms=resize)
        self.eval_dataloader = DataLoader(eval_dataset, config.batch_size, False, num_workers=config.num_workers, pin_memory=True)

        logging.info(f'dataset | folder: {str(folder)} |   val size: {len(self.eval_dataloader) * config.batch_size}')

    @torch.no_grad()
    def eval_generator(self, input: Tensor) -> dict:
        """
        Eval generator
        """

        logging.debug('eval generator')
        
        input_1, input_b_1, input_b_2 = self.net_ext(input)
        input_att = self.net_att(input)
        fus_1 = input_1 * input_att
        fus_2 = self.net_con(fus_1, input_b_1, input_b_2)

        # calculate loss towards criterion
        gamma = 1

        l_mse = self.mse(fus_2, input).mean() ** 0.5
        l_ssim = self.ssim(fus_2, input).mean() * 2

        l_total = l_mse + gamma * l_ssim

        return l_total.item()

    def __call__(self):   
        # Eval Generator
        eval_process = tqdm(enumerate(self.eval_dataloader), total=len(self.eval_dataloader))
        total_loss = 0
        for idx, eval_sample in eval_process:
            eval_im = eval_sample['im']
            eval_im = eval_im.to(self.device)

            loss = self.eval_generator(eval_im)
            total_loss += loss
        logging.info(f'total_loss: {total_loss:03f}')


def parse_args() -> Namespace:
    # args parser
    parser = argparse.ArgumentParser()

    # universal opt
    parser.add_argument('--id', default='a8', help='train process identifier')
    parser.add_argument('--folder', default='../datasets/FLIR_ADAS_v2', help='data root path')
    parser.add_argument('--size', default=256, help='resize image to the specified size')

    # dataloader opt
    parser.add_argument('--batch_size', type=int, default=16, help='dataloader batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='dataloader workers number')

    return parser.parse_args()


if __name__ == '__main__':
    config = parse_args()
    logging.basicConfig(level='INFO')
    
    model = config.id
    # eval = Eval(f'./cache/{model}/best.pth', config)
    eval = Eval(f'./weights/default.pth', config)
    eval()