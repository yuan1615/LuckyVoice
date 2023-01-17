import os
import torch
from pprint import pprint
import numpy as np
import sys
from scipy.io import wavfile
import argparse
from tqdm import tqdm

sys.path.append('./dataprocessing/vad')
from silerovad.utils_vad import *

SAMPLING_RATE = 16000
torch.set_num_threads(1)

model, utils = torch.hub.load(repo_or_dir='./dataprocessing/vad/silerovad',
                              source='local',
                              model='silero_vad',
                              force_reload=True,
                              onnx=False)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils


def main(pth, save_pth):
    os.makedirs(save_pth, exist_ok=True)
    names = os.listdir(pth)
    for name in tqdm(names):
        if '.wav' in name:
            wav = read_audio(os.path.join(pth, name), sampling_rate=SAMPLING_RATE)
            wav_np = wav.numpy()
            # get speech timestamps from full audio file
            speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE, threshold=0.8)
            # 保存切分的片段
            i = 1
            for d in speech_timestamps:
                start = d['start']
                end = d['end']
                if (end - start)/SAMPLING_RATE < 3.0:
                    continue
                wav_np_temp = wav_np[start:end] * 32767.0
                wavfile.write(
                    os.path.join(save_pth, name.split('.')[0] + '_cut_' + str(i) + '.wav'),
                    SAMPLING_RATE,
                    wav_np_temp.astype(np.int16),
                )
                i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', default='/home/admin/yuanxin/LuckyData/bilibili')
    parser.add_argument('--save_pth', default='/home/admin/yuanxin/LuckyData/bilibili_vad')
    a = parser.parse_args()
    main(a.pth, a.save_pth)

