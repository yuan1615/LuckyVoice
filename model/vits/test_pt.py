import os

import torch
from tqdm import tqdm

path = '/home/admin/yuanxin/vits/DUMMY3'

filenames = os.listdir(path)

for file in tqdm(filenames):
    if '.spec.pt' in file:
        spec = torch.load(os.path.join(path, file))



