"""
Pytorch implement of Rad-cGAN v1.0

Rad-cGAN v1.0: Radar-based precipitation nowcasting model with conditional Generative Adversarial Networks for multiple dam domains

https://doi.org/10.5194/gmd-15-5967-2022

https://zenodo.org/record/6650722#.ZD0Nh3ZBwuU


"""  
import torch
from torch import nn
class CBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm="bnorm", relu=0.0):
        super().__init__()

        layers = []
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, stride=stride, padding=padding,
                             bias=bias)]

        if not norm is None:
            if norm == "bnorm":
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif norm == "inorm":
                layers += [nn.InstanceNorm2d(num_features=out_channels)]

        if not relu is None and relu >= 0.0:
            layers += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)
    

    
class Rad_cGAN_generator(nn.Module):
    def __init__(self, in_channels, out_channels=1, nker=64, norm="bnorm"):
        super(Rad_cGAN_generator, self).__init__()


        self.enc1 = CBR2d(in_channels=in_channels, out_channels=2 * nker, norm=norm, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.drop1 = nn.Dropout(p=0.5)

        self.enc2_1 = CBR2d(in_channels=2 * nker, out_channels=4 * nker, norm=norm, padding=1)
        self.enc2_2 = CBR2d(in_channels=4 * nker, out_channels=8 * nker, norm=norm, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.drop2 = nn.Dropout(p=0.5)

        self.enc3 = CBR2d(in_channels=8 * nker, out_channels=16 * nker, norm=norm, padding=1)
        self.drop3 = nn.Dropout(p=0.5)
        self.upsample3 = nn.Upsample(scale_factor=2)
        

        self.enc4_1 = CBR2d(in_channels=24 * nker, out_channels=8 * nker, norm=norm, padding=1)
        self.drop4 = nn.Dropout(p=0.5)
        self.enc4_2 = CBR2d(in_channels=8 * nker, out_channels=4 * nker, norm=norm, padding=1)
        self.upsample4 = nn.Upsample(scale_factor=2)

        self.enc5_1 = CBR2d(in_channels=6 * nker, out_channels=2 * nker, norm=norm, padding=1)
        self.drop5_1 = nn.Dropout(p=0.5)

        self.enc5_2 = CBR2d(in_channels=2 * nker, out_channels=1 * nker, norm=norm, padding=1)
        self.drop5_2 = nn.Dropout(p=0.5)

        self.enc5_3 = CBR2d(in_channels=1 * nker, out_channels=2 * out_channels, norm=norm, padding=1)

        self.fc = nn.Conv2d(in_channels=2 * out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):

        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        drop1 = self.drop1(pool1)

        enc2_1 = self.enc2_1(drop1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)
        drop2 = self.drop2(pool2)

        enc3 = self.enc3(drop2)
        drop3 = self.drop3(enc3)
        upsample3 = self.upsample3(drop3)
        cat3 = torch.cat((upsample3,enc2_2), dim=1)

        enc4_1 = self.enc4_1(cat3)
        drop4 = self.drop4(enc4_1)
        
        enc4_2 = self.enc4_2(drop4)
        upsample4 = self.upsample4(enc4_2)
        cat4 = torch.cat((upsample4,enc1), dim=1)
        
        enc5_1 = self.enc5_1(cat4)
        drop5_1 = self.drop5_1(enc5_1)

        enc5_2 = self.enc5_2(drop5_1)
        drop5_2 = self.drop5_2(enc5_2)

        enc5_3 = self.enc5_3(drop5_2)

        x = self.fc(enc5_3)


        return x

class Rad_cGAN_discriminator(nn.Module):
    
    
    def __init__(self, in_channels, out_channels=1, nker=64, norm="bnorm"):
        super(Rad_cGAN_discriminator, self).__init__()

        self.enc1 = CBR2d(in_channels= in_channels, out_channels=1 * nker, kernel_size=4, stride=2, norm=norm, relu=0.2, padding=0)

        self.enc2 = CBR2d(in_channels=1 *  nker, out_channels=2 * nker, kernel_size=4, stride=2, norm=norm, relu=0.2, padding=0)

        self.enc3 = CBR2d(in_channels=2 * nker, out_channels=4 * nker, kernel_size=4, stride=1, norm=norm, relu=0.2, padding=0)

        self.fc = nn.Conv2d(in_channels=4 * nker, out_channels=out_channels, kernel_size=4, stride=1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        x = self.fc(x)
        
        x = torch.sigmoid(x)

        return x


