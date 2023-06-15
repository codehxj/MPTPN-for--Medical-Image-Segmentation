# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from nets.LViT_encoder import LViT_encoder
import Config_Natural

config_vit = Config_Natural.get_CTranS_config()


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        self.encoder = LViT_encoder(config_vit, n_channels=3, n_classes=1)
        self.projector = nn.AdaptiveAvgPool2d((1,1))
        self.predictor1 = nn.Sequential(nn.Linear(64, 64, bias=False),
                                        #nn.BatchNorm1d(64),
                                        nn.LeakyReLU(inplace=True),  # hidden layer
                                        nn.Linear(64, 64))  # output layer
        self.predictor2 = nn.Sequential(nn.Linear(128, 128, bias=False),
                                        #nn.BatchNorm1d(128),
                                        nn.LeakyReLU(inplace=True),  # hidden layer
                                        nn.Linear(128, 128))  # output layer
        self.predictor3 = nn.Sequential(nn.Linear(256, 256, bias=False),
                                        #nn.BatchNorm1d(256),
                                        nn.LeakyReLU(inplace=True),  # hidden layer
                                        nn.Linear(256, 256))  # output layer

    def forward(self, x1, x2, text):
        # ==============修改代码==================================x1[2,3,224,224] text[2,10,768]
        x1_1, x1_2, x1_3 = self.encoder(x1, text)  #[1,64,224,224] [1,128,112,112] [1,256,56,56]
        x2_1, x2_2, x2_3 = self.encoder(x2, text)  #[1,64,224,224] [1,128,112,112] [1,256,56,56]

        x1_1 = self.projector(x1_1) #[1,64,1,1]
        x1_1 = torch.flatten(x1_1, 1)
        x1_2 = self.projector(x1_2) #[1,128,1,1]
        x1_2 = torch.flatten(x1_2, 1)
        x1_3 = self.projector(x1_3) #[1,256,1,1]
        x1_3 = torch.flatten(x1_3, 1)
        x2_1 = self.projector(x2_1) #[1,64,1,1]
        x2_1 = torch.flatten(x2_1, 1)
        x2_2 = self.projector(x2_2) #[1,128,1,1]
        x2_2 = torch.flatten(x2_2, 1)
        x2_3 = self.projector(x2_3) #[1,256,1,1]
        x2_3 = torch.flatten(x2_3, 1)

        p1_1 = self.predictor1(x1_1)  # NxC [16,512]
        p1_2 = self.predictor2(x1_2)
        p1_3 = self.predictor3(x1_3)
        p2_1 = self.predictor1(x2_1)  # NxC [16,512]
        p2_2 = self.predictor2(x2_2)
        p2_3 = self.predictor3(x2_3)

        return p1_1, p1_2, p1_3, p2_1, p2_2, p2_3, x1_1.detach(), x1_2.detach(), x1_3.detach(), x2_1.detach(), x2_2.detach(), x2_3.detach()
        # ======================================================