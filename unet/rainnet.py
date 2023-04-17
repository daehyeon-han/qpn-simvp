"""
RainNet v1.0: a convolutional neural network for radar-based precipitation nowcasting

https://doi.org/10.5194/gmd-13-2631-2020

https://github.com/hydrogo/rainnet

"""
import torch
from torch import nn

class RainNet(nn.Module):
    def __init__(self, shape_in,  
                 kernel_size = 3,
                 mode = "regression",
                 conv_shape = [["2" , [64,128]],
                        ["3" , [128,256]],
                        ["4" , [256,512]],
                        ["5" , [512,1024]],
                        ["6" , [1536,512]],
                        ["7" , [768,256]],
                        ["8" , [384,128]],
                        ["9" , [192,64]]]):
        
        super().__init__()
        T, C, H, W = shape_in
        self.kernel_size = kernel_size
        self.mode = mode
        self.im_shape = [H,W]
        #self.im_shape = (512,512),
        conv_shape.insert(0, ["1", [T,64]])

        self.conv = nn.ModuleDict()
        for name, (in_ch, out_ch) in conv_shape:
            if name != "9":
                self.conv[name] = self.make_conv_block(in_ch,
                                                       out_ch ,self.kernel_size)
            else:
                self.conv[name] = nn.Sequential(
                    self.make_conv_block(in_ch, out_ch, self.kernel_size),
                    nn.Conv2d(out_ch, 2, 3, padding='same'))
        
        
        self.pool = nn.MaxPool2d(kernel_size = (2,2))
        self.upsample = nn.Upsample(scale_factor=(2,2))
        self.drop = nn.Dropout(p=0.5)
        #Ayzel et al. uses basic dropout
        ##self.drop = nn.Dropout2d(p=0.5)
        
        self.last_layer = nn.Sequential(
            nn.Conv2d(2, T, kernel_size=1, padding = 'valid'),
            # MISTAKE : not a fully connected layer, but linear activation (no activation)
            #nn.Linear(self.im_shape[0], self.im_shape[1])
            )

            
    def make_conv_block(self, in_ch, out_ch, kernel_size):
        
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding='same'),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding='same'),
            nn.ReLU()
            )
    
        
    def forward(self, x):
        x=x[:,:,0] # Reduce channel dimension
        x1s = self.conv["1"](x.float()) # conv1s
        x2s = self.conv["2"](self.pool(x1s)) # conv2s
        x3s = self.conv["3"](self.pool(x2s)) # conv3s
        x4s = self.conv["4"](self.pool(x3s)) # conv4s
        x = self.conv["5"](self.pool(self.drop(x4s))) # conv5s
        x = torch.cat((self.upsample(self.drop(x)), x4s), dim=1) # up6
        x = torch.cat((self.upsample(self.conv["6"](x)), x3s), dim=1) # up7
        x = torch.cat((self.upsample(self.conv["7"](x)), x2s), dim=1) # up8
        x = torch.cat((self.upsample(self.conv["8"](x)), x1s), dim=1) # up9
        x = self.conv["9"](x) #conv9
        x = self.last_layer(x) #outputs
        
        x = torch.unsqueeze(x,2) # Add channel dimension
        
        return x
