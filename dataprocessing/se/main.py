import os
import argparse
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model='./ckpt/damo/speech_frcrn_ans_cirm_16k')


def main(pth, save_pth):
    files = os.listdir(pth)
    os.makedirs(save_pth, exist_ok=True)
    for f in files:
        if '.wav' in f:
            result = ans(
                os.path.join(pth, f),
                output_path=os.path.join(save_pth, 'clear_' + f.split('.')[0] + '.wav'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', default='../LuckyData/bilibili_vad')
    parser.add_argument('--save_pth', default='../LuckyData/bilibili_vad_clear')
    a = parser.parse_args()
    main(a.pth, a.save_pth)

