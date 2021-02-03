import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import transforms

import os
import sys
sys.path.append(os.getcwd())

class MinusChecker(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(2, 2), stride=2, padding=1),
            nn.BatchNorm2d(16), 
            nn.LeakyReLU(0.05),
        
            nn.Conv2d(16, 32, kernel_size=(2, 2), stride=2, padding=1),
            nn.BatchNorm2d(32), 
            nn.LeakyReLU(0.05),

            nn.Conv2d(32, 64, kernel_size=(2, 2), stride=2, padding=1),
            nn.BatchNorm2d(64), 
            nn.LeakyReLU(0.05),
        )

        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv(x)
        x = x.mean([2, 3])
        x = self.fc(x)
        return nn.Sigmoid()(x)

model = MinusChecker()
model.load_state_dict(torch.load('./checkpoints_mlt/minus_checker.pth'))

i = 0
images = []
while True:
    try:
        img = Image.open('./data/cropped/' + str(i) + '.png').convert('RGB')
        images.append(img)
    except:
        break
    i += 1

with open('./data/res/minuses.txt', 'w+') as f:
    for img in images:

        x = transforms.PILToTensor()(img).unsqueeze(0) / 255
        res = model(x)[:, 1]

        if res[0] > 0.5:
            f.write('1')
        else:
            f.write('-1')
        f.write('\n')





