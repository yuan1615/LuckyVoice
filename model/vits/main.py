import argparse
from models import SynthesizerTrn
from text.symbols import symbols
import utils
import ttsfrd
from scipy.io import wavfile
import numpy as np
from synthesize_fastapi import *
config = "./model/vits/configs/baker_base.json"

print("---------- Loading VITS Model ----------")
hps = utils.get_hparams_from_file(config)
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint("./ckpt/vits/pretrained_baker.pth", net_g, None)

lexicon_mandarin = read_lexicon("./model/vits/lexicon/pinyin-lexicon-r.txt")

frontend = ttsfrd.TtsFrontendEngine()
model_dir = './ckpt/damo/resource'
frontend.initialize(model_dir)
frontend.set_lang_type('zhcn')


def tts(text, out):
    audio_all = np.zeros(1, dtype=np.int16)  # 设置初始音频
    pinyin_list, prosody_list = g2p_mandarin(frontend, text)
    for texts, phone_prosody in zip(pinyin_list, prosody_list):
        stn_tst = get_text(texts, hps)
        prosody = get_prosody(phone_prosody, hps)
        with torch.no_grad():
            x_tst = stn_tst.cuda().unsqueeze(0)
            prosody = prosody.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            audio = net_g.infer(x_tst, x_tst_lengths, prosody, noise_scale=.667, noise_scale_w=0.0, length_scale=1)[0][
                        0, 0].data.cpu().float().numpy() * 32767.0
        i = np.random.uniform(0.12, 0.25, 1)[0]
        space_time = np.zeros(int(i * 22050), dtype=np.int16)
        audio_all = np.concatenate((audio_all, audio, space_time))
    wavfile.write(
        out,
        22050,
        audio_all.astype(np.int16),
    )
    return 'OK'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default='我说点什么好呢？念一个绕口令吧。八百标兵奔北坡，炮兵并排北边跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。')
    parser.add_argument('--out', default='output.wav')
    a = parser.parse_args()
    tts(a.text, a.out)
