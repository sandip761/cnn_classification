import torch
import torch.nn as nn

class YOLO(nn.Module):
    def __init__(self):
        super(YOLO, self).__init__()
        self.conv1_7_2 = nn.Conv2d(3, 64, 7, stride=2, padding=(4, 4))
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.maxpool1_2_2  = nn.MaxPool2d(2, stride=2, padding=0)
        #torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

        self.conv1_3_2 = nn.Conv2d(64, 192, 3, stride=1, padding=1)
    def forward(self, x):
        out = self.conv1_7_2(x)
        out = self.maxpool1_2_2(out)
        out = self.conv1_3_2(out)
        out = self.maxpool1_2_2(out)
        return out

img = torch.randn(1,3,448,448)
model = YOLO()
x = model(img)
print(x.size())