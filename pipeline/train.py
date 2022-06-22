import logging
from functools import reduce
from pathlib import Path

import torch

from kornia.losses import SSIMLoss
from kornia.metrics import AverageMeter
from torch import nn, Tensor
from torch.optim import RMSprop
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from functions.div_loss import div_loss

from models.attention import Attention
from models.constructor import Constructor
from models.extractor import Extractor

from modules.discriminator import Discriminator

from utils.environment_probe import EnvironmentProbe
from utils.fusion_data import FusionData


class Train:
    """
    The train process for TarDAL.
    """

    def __init__(self, environment_probe: EnvironmentProbe, config: dict):
        logging.info(f'Training')
        self.config = config
        self.environment_probe = environment_probe

        # modules
        self.net_ext = Extractor()
        self.net_con = Constructor()
        self.net_att = Attention()

        self.discriminator = Discriminator()

        # WGAN adam optim
        logging.info(f'RMSprop | learning rate: {config.learning_rate}')
        self.opt_net_ext = RMSprop(self.net_ext.parameters(), lr=config.learning_rate)
        self.opt_net_con = RMSprop(self.net_con.parameters(), lr=config.learning_rate)
        self.opt_net_att = RMSprop(self.net_att.parameters(), lr=config.learning_rate)
        self.opt_discriminator = RMSprop(self.discriminator.parameters(), lr=config.learning_rate)

        # move to device
        logging.info(f'module device: {environment_probe.device}')

        self.net_ext.to(environment_probe.device)
        self.net_con.to(environment_probe.device)
        self.net_att.to(environment_probe.device)
        self.discriminator.to(environment_probe.device)

        # loss
        self.mse = nn.MSELoss(reduction='none')
        self.ssim = SSIMLoss(window_size=11, reduction='none')
        self.mse.cuda()
        self.ssim.cuda()

        # WGAN div hyper parameters
        # self.wk, self.wp = 2, 6

        # datasets
        folder = Path(config.folder)
        resize = transforms.Resize((config.size, config.size))
        train_dataset = FusionData(folder, mode='train', transforms=resize)
        self.train_dataloader = DataLoader(train_dataset, config.batch_size, True, num_workers=config.num_workers, pin_memory=True)

        eval_dataset = FusionData(folder, mode='eval', transforms=resize)
        self.eval_dataloader = DataLoader(eval_dataset, config.batch_size, False, num_workers=config.num_workers, pin_memory=True)

        logging.info(f'dataset | folder: {str(folder)} | train size: {len(self.train_dataloader) * config.batch_size}')
        logging.info(f'dataset | folder: {str(folder)} |   val size: {len(self.eval_dataloader) * config.batch_size}')


    def train_discriminator(self, input: Tensor) -> Tensor:
        """
        Train discriminator
        """

        logging.debug('train discriminator')
        # switch to train mode
        self.discriminator.train()

        # sample real & fake
        real_s = input
        self.net_ext.eval()
        self.net_con.eval()
        self.net_att.eval()

        input_1, input_b_1, input_b_2 = self.net_ext(input)
        input_att = self.net_att(input)
        fus_1 = input_1 * input_att
        fake_s = self.net_con(fus_1, input_b_1, input_b_2)

        # judge value towards real & fake
        real_v = torch.squeeze(self.discriminator(real_s))
        fake_v = torch.squeeze(self.discriminator(fake_s))

        # loss calculate
        real_l, fake_l = -real_v.mean(), fake_v.mean()
        # div = div_loss(self.discriminator, real_s, fake_s, self.wp)
        # loss = real_l + fake_l + self.wk * div

        loss = real_l + fake_l


        # backwardD
        self.opt_discriminator.zero_grad()
        loss.backward()
        self.opt_discriminator.step()

        return loss.item()

    def train_generator(self, input: Tensor) -> dict:
        """
        Train generator 'ir + vi -> fus'
        """

        logging.debug('train generator')
        
        self.net_ext.train()
        self.net_con.train()
        self.net_att.train()

        input_1, input_b_1, input_b_2 = self.net_ext(input)
        input_att = self.net_att(input)
        fus_1 = input_1 * input_att
        fuse = self.net_con(fus_1, input_b_1, input_b_2)

        # calculate loss towards criterion
        gamma = 1
        l_mse = self.mse(fuse, input).mean() ** 0.5
        l_ssim = self.ssim(fuse, input).mean() * 2
        l_total = l_mse + gamma * l_ssim

        self.discriminator.eval()
        # l_adv = -self.discriminator(fuse).mean()  # judge target: fus

        D1_loss_real = - (torch.log(self.discriminator(input))).mean()
        D1_loss_fake = - (torch.log(1. - self.discriminator(fuse))).mean()
        l_adv = D1_loss_fake + D1_loss_real

        loss = l_total + l_adv
        # loss = l_total

        # backwardG
        self.opt_net_ext.zero_grad()
        self.opt_net_con.zero_grad()
        self.opt_net_att.zero_grad()
        loss.backward()
        self.opt_net_ext.step()
        self.opt_net_con.step()
        self.opt_net_att.step()

        state = {
            'g_loss': loss.item(),
            'g_l_total': l_total.item(),
            'g_l_dis': l_adv.item(),
        }
        return state

    def eval_generator(self, input: Tensor) -> dict:
        """
        Eval generator
        """

        logging.debug('eval generator')
        
        self.net_ext.eval()
        self.net_con.eval()
        self.net_att.eval()

        input_1, input_b_1, input_b_2 = self.net_ext(input)
        input_att = self.net_att(input)
        fus_1 = input_1 * input_att
        fuse = self.net_con(fus_1, input_b_1, input_b_2)

        # calculate loss towards criterion
        gamma = 1

        l_mse = self.mse(fuse, input).mean() ** 0.5
        l_ssim = self.ssim(fuse, input).mean() * 2

        l_total = l_mse + gamma * l_ssim

        loss = l_total

        return loss.item()

    def run(self):   
        best = float('Inf')
        best_epoch = 0

        for epoch in range(1, self.config.epochs + 1):
            train_process = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
            
            meter = AverageMeter()
            for idx, train_sample in train_process:
                
                train_im = train_sample['im']
                train_im = train_im.to(self.environment_probe.device)

                g_loss = self.train_generator(train_im)
                # d_loss = 0
                d_loss = self.train_discriminator(train_im)

                meter.update(Tensor(list(g_loss.values()) + [d_loss]))

            # Eval Generator
            total_loss = 0
            eval_process = tqdm(enumerate(self.eval_dataloader), total=len(self.eval_dataloader))
            for idx, eval_sample in eval_process:
                eval_im = eval_sample['im']
                eval_im = eval_im.to(self.environment_probe.device)

                loss = self.eval_generator(eval_im)
                total_loss += loss

            logging.info(f'[{epoch}] g_loss: {meter.avg[0]:03f} | g_l_total: {meter.avg[1]:03f} | g_l_dis: {meter.avg[2]:03f} | d_loss: {meter.avg[3]:03f} | total_loss: {total_loss:03f}')

            if epoch % 1 == 0:

                if best > meter.avg[0]:
                    best = meter.avg[0]
                    best_epoch = epoch

                    self.save(epoch, is_best=True)

                self.save(epoch)
                logging.info(f'best epoch is {best_epoch}, loss is {best}')



    def save(self, epoch: int, is_best = False):
        path = Path(self.config.cache) / self.config.id
        path.mkdir(parents=True, exist_ok=True)

        if is_best:
            cache = path / f'best.pth'
        else:
            cache = path / f'{epoch:03d}.pth'
        logging.info(f'save checkpoint to {str(cache)}')

        state = {
            'ext': self.net_ext.state_dict(),
            'con': self.net_con.state_dict(),
            'att': self.net_att.state_dict(),
            'd': self.discriminator.state_dict(),

            'opt': {
                'ext': self.opt_net_ext.state_dict(),
                'con': self.opt_net_con.state_dict(),
                'att': self.opt_net_att.state_dict(),
                'd': self.opt_discriminator.state_dict(),
            },
        }

        torch.save(state, cache)
