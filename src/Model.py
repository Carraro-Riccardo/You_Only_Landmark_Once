import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            conv3x3(in_ch, out_ch, stride=stride),
            nn.LeakyReLU(0.2, inplace=True),
            conv3x3(out_ch, out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x): return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False) #or bilinear
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UpLearn(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch, ch, 3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x): return self.up(x)

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=True),
        )
    def forward(self, x): return x + self.body(x)

class SuperResolutionUNet(nn.Module):
    """
    Efficient U-Net SR (16x16 → 128x128)
    - Strided convs instead of MaxPool
    - align_corners=False
    - 3x UpLearn (nearest+conv)
    - Small refine head at 128x
    - Optional heatmap injection at refine stage
    """
    def __init__(self, in_channels=3, base_filters=32, out_channels=3, refine_blocks=3):
        super().__init__()

        # Encoder (strided)
        self.enc1 = DoubleConv(in_channels,       base_filters,   stride=1)  # 16x
        self.enc2 = DoubleConv(base_filters,      base_filters*2, stride=2)  # 8x
        self.enc3 = DoubleConv(base_filters*2,    base_filters*4, stride=2)  # 4x
        self.enc4 = DoubleConv(base_filters*4,    base_filters*8, stride=2)  # 2x bottleneck in size after next

        # Bottleneck
        self.bottleneck = DoubleConv(base_filters*8, base_filters*8)

        # Decoder (back to 16x)
        self.up3 = UpBlock(base_filters*8, base_filters*4, base_filters*4)  # 2x->4x, skip e3
        self.up2 = UpBlock(base_filters*4, base_filters*2, base_filters*2)  # 4x->8x, skip e2
        self.up1 = UpBlock(base_filters*2, base_filters,   base_filters)    # 8x->16x, skip e1

        # Learned upsampling to 128x
        self.up_learn1 = UpLearn(base_filters)  # 16->32
        self.up_learn2 = UpLearn(base_filters)  # 32->64
        self.up_learn3 = UpLearn(base_filters)  # 64->128

        # Refine head
        refine_in = base_filters
        self.refine_in = nn.Conv2d(refine_in, base_filters, 1)
        self.refine = nn.Sequential(*[ResBlock(base_filters) for _ in range(refine_blocks)])

        # Final projection + global skip
        self.final_conv = nn.Conv2d(base_filters, out_channels, 1)

    def forward(self, x):
        # encoder
        e1 = self.enc1(x)        # 16x
        e2 = self.enc2(e1)       # 8x
        e3 = self.enc3(e2)       # 4x
        e4 = self.enc4(e3)       # 2x

        b  = self.bottleneck(e4)

        # decoder to 16x
        d3 = self.up3(b, e3)     # 4x
        d2 = self.up2(d3, e2)    # 8x
        d1 = self.up1(d2, e1)    # 16x

        # learned upsampling to 128x
        u1 = self.up_learn1(d1)  # 32x
        u2 = self.up_learn2(u1)  # 64x
        u3 = self.up_learn3(u2)  # 128x

        r = self.refine(self.refine_in(u3))
        out = self.final_conv(r)

        up_input = F.interpolate(x, size=out.shape[2:], mode='bicubic', align_corners=False) #or bilinear
        return out + up_input
