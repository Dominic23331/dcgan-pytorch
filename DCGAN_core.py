import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.nn import BCELoss
from torch.utils.data import DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchsummary import summary
import pandas as pd
from tqdm import tqdm

from model.gan_model import Discriminator, Generator
from utils.dataloader import DCGAN_dataloader, transform


class DCGAN:
    _defaults = {
        "lr_d": 0.001,
        "lr_g": 0.001,
        "nz": 128,
        "nc": 3,
        "ngf": 64,
        "ndf": 64,
        "device": "cuda",
        "model_path": "你的模型文件路径"
    }

    def __init__(self):
        self.__dict__.update(self._defaults)
        self.Discriminator = Discriminator(self.nc, self.ndf)
        self.Generator = Generator(self.nz, self.nc, self.ngf)
        if self.device == "cuda":
            self.Generator.cuda()
            self.Discriminator.cuda()

    def summary(self, net):
        if net == "Generator":
            summary(self.Generator, (self.nz, 1, 1))
        elif net == "Discriminator":
            summary(self.Discriminator, (3, 64, 64))
        else:
            raise KeyError("请输入Generator或Discriminator")

    def train(self, epoch=20, bach_size=64):
        dataset = DCGAN_dataloader("./dataset/data.csv", transforms=transform())
        data = DataLoader(
            dataset,
            batch_size=bach_size,
            shuffle=True,
            drop_last=True
        )

        optim_D = Adam(
            self.Discriminator.parameters(),
            lr=self.lr_d,
            betas=[0.5, 0.999]
        )
        optim_G = Adam(
            self.Generator.parameters(),
            lr=self.lr_g,
            betas=[0.5, 0.999]
        )

        loss_func = BCELoss()

        loss_g_log = []
        loss_d_log = []
        loss_d = 0
        loss_g = 0
        print("#" * 10 + " 开始训练 " + "#" * 10)
        with tqdm(total=epoch, desc="训练进度", postfix=dict, mininterval=0.3) as pbar:
            for i_epoch in range(epoch):
                loss_d = 0
                loss_g = 0
                for i, batch_data in enumerate(data):
                    img = batch_data
                    img = Variable(img)
                    if self.device == "cuda":
                        img = img.cuda()
                    self.Discriminator.zero_grad()
                    label = torch.ones((bach_size,), device=self.device)
                    out = self.Discriminator(img).view(-1)
                    errD_real = loss_func(out, label)
                    errD_real.backward()
                    D_x = out.mean().item()

                    noise = torch.randn(bach_size, self.nz, 1, 1, device=self.device)
                    fake = self.Generator(noise)
                    label.fill_(0)
                    out = self.Discriminator(fake.detach()).view(-1)
                    errD_fake = loss_func(out, label)
                    errD_fake.backward()
                    D_G_z1 = out.mean().item()
                    errD = errD_fake + errD_real
                    loss_d += errD.item()
                    optim_D.step()

                    self.Generator.zero_grad()
                    label.fill_(1)
                    out = self.Discriminator(fake).view(-1)
                    errG = loss_func(out, label)
                    errG.backward()
                    loss_g += errG.item()
                    optim_G.step()

                    if i % 5 == 0:
                        pbar.set_postfix(**{'loss_D': "{:.4f}".format(loss_d / (i + 1)),
                                            'loss_G': "{:.4f}".format(loss_g / (i + 1)),
                                            'D(x)': "{:.4f}".format(D_x),
                                            'D(G(z))': "{:.4f}".format(D_G_z1)
                                            })
                        loss_g_log.append(loss_g / (i + 1))
                        loss_d_log.append(loss_d / (i + 1))
                with torch.no_grad():
                    img = self.Generator(torch.randn(20, self.nz, 1, 1, device=self.device))
                img = img * 255
                for j in range(20):
                    img = np.array(img[j].cpu()).transpose((1, 2, 0))
                    cv2.imwrite("/log/img_log/epoch{}/img{}.jpg".format(i_epoch, j), img)
                pbar.update(1)
                torch.save(self.Generator.state_dict(),
                           "./log/model_g/epoch{}_loss{}.pth".format(i_epoch, loss_g / len(loss_g_log)))
                torch.save(self.Discriminator.state_dict(),
                           "./log/model_d/epoch{}_loss{}.pth".format(i_epoch, loss_d / len(loss_d_log)))
        torch.cuda.empty_cache()
        loss_log = pd.DataFrame()
        loss_log["g_loss"] = loss_g_log
        loss_log["d_loss"] = loss_d_log
        loss_log.to_csv("./log/loss/loss.csv")

    def draw_loss(self):
        loss = pd.read_csv("./log/loss/loss.csv")
        plt.plot(loss["g_loss"], color="orange", label="Generator")
        plt.plot(loss["d_loss"], color="red", label="Discriminitor")
        plt.legend(["Generator", "Discriminitor"])
        plt.show()

    def generate(self, path="./img/generate.jpg"):
        param = torch.load(self.model_path)
        self.Generator.load_state_dict(param)
        with torch.no_grad():
            img = self.Generator(torch.randn(1, self.nz, 1, 1, device=self.device))
        img = img * 255
        img = np.array(img[0].cpu()).transpose((1, 2, 0))
        cv2.imshow("generate", img)
        cv2.waitKey(0)
        cv2.imwrite(path, img)


if __name__ == '__main__':
    core = DCGAN()
    core.draw_loss()
