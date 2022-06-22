import torch
import torch.nn as nn
from torch import Tensor

from models.attention import Attention
from models.constructor import Constructor
from models.extractor import Extractor

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # extractor
        self.net_ext = Extractor()
        # constructor
        self.net_con = Constructor()
        # attention
        self.net_att = Attention()
        # softmax and feather-fuse
        # self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input: Tensor) -> Tensor:
        input_1, input_b_1, input_b_2 = self.net_ext(input)

        input_att = self.net_att(input)

        fus_1 = input_1 * input_att
        # fus_1 = self.softmax(fus_1)

        fus_2 = self.net_con(fus_1, input_b_1, input_b_2)

        return fus_2
