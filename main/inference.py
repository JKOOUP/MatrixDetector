import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.autograd import Variable

import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import tools.utils as utils
import tools.dataset as dataset
from models.moran import MORAN


class Recognizer:
    def __init__(self, model_path):
        alphabet = '0:1:2:3:4:5:6:7:8:9:a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w:x:y:z:$'

        self.cuda_flag = torch.cuda.is_available()
        if self.cuda_flag:
            self.MORAN = MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True, CUDA=self.cuda_flag)
            self.MORAN = self.MORAN.cuda()
        else:
            self.MORAN = MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True, inputDataType='torch.FloatTensor', CUDA=self.cuda_flag)

        print('loading pretrained model from %s' % model_path)
        if self.cuda_flag:
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location='cpu')

        MORAN_state_dict_rename = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            MORAN_state_dict_rename[name] = v
        self.MORAN.load_state_dict(MORAN_state_dict_rename)

        for p in self.MORAN.parameters():
            p.requires_grad = False
        self.MORAN.eval()

        self.converter = utils.strLabelConverterForAttention(alphabet, ':')
        self.transformer = dataset.resizeNormalize((100, 32))

    def preprocess(self, img):
        image = Image.fromarray(img[..., ::-1]).convert('L')
        image = self.transformer(image)
        image = image.view(1, *image.size())
        return image

    def predict(self, img_batch):
        batch_size = int(img_batch.size(0))

        if self.cuda_flag:
            img_batch = img_batch.cuda()

        text = torch.LongTensor(batch_size * 5)
        length = torch.IntTensor(batch_size)

        max_iter = 20
        t, l = self.converter.encode(['0' * max_iter] * batch_size)
        utils.loadData(text, t)
        utils.loadData(length, l)

        output = self.MORAN(img_batch, length, text, text, test=True, debug=True)
        return output, length

    def post_process(self, output, length):
        preds, preds_reverse = output[0]

        n_preds = torch.zeros(preds.shape[0], dtype=torch.int)
        for i, p in enumerate(preds):
            if p.argmax() == 36:
                n_preds[i] = 36
            else:
                n_preds[i] = int(p[:10].argmax())

        _, preds = preds.max(1)
        _, preds_reverse = preds_reverse.max(1)

        sim_preds = self.converter.decode(n_preds.data, length.data)
        sim_preds = list(map(lambda x: x.strip().split('$')[0], sim_preds))
        sim_preds_reverse = self.converter.decode(preds_reverse.data, length.data)
        sim_preds_reverse = list(map(lambda x: x.strip().split('$')[0], sim_preds_reverse))
        return sim_preds, sim_preds_reverse

    def __call__(self, images):
        unit_size = len(images) == 1
        if unit_size:
            images = images * 2

        img_tensors = []
        for img in images:
            img_tensors.append(self.preprocess(img))

        img_batch = torch.cat(img_tensors)

        output, length = self.predict(img_batch)

        sim_preds, sim_preds_reverse = self.post_process(output, length)

        if unit_size:
            sim_preds = sim_preds[:1]

        return sim_preds

rec = Recognizer('./checkpoints_mlt/BestMoran.pth')

i = 0
images = []
while True:
    try:
        img = np.array(Image.open('./data/cropped/' + str(i) + '.png').convert('RGB'))
        images.append(img)
    except:
        break
    i += 1

result = rec(images)

w = []
with open('./data/res/size.txt', 'r') as f:
    h = int(f.readline())
    for i in range(h):
        w.append(int(f.readline()))

idx = 0
for i in range(len(result)):
    print(result[i], end=' ')
    w[idx] -= 1
    if w[idx] == 0:
        print()
        idx += 1
