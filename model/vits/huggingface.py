import argparse
import re
from models import SynthesizerTrn
from text.symbols import symbols
import utils
import ttsfrd
import gradio as gr
import tempfile
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


def tts(text):
    text = re.sub('[a-zA-Z]', '', text)
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
        i = np.random.uniform(0.12, 0.35, 1)[0]
        space_time = np.zeros(int(i * 22050), dtype=np.int16)
        audio_all = np.concatenate((audio_all, audio, space_time))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        wavfile.write(
            fp.name,
            22050,
            audio_all.astype(np.int16),
        )
    return fp.name


if __name__ == '__main__':
    inputs = [gr.inputs.Textbox(label="Input Text", default='祝大家中秋节快乐', lines=6)]
    outputs = gr.Audio(label="Output")
    interface = gr.Interface(fn=tts, inputs=inputs, outputs=outputs,
                             examples=[['目前申请好人贷支持身份证原件实时拍摄或上传相册照片两种方式，但复印件及临时身份证是不可以的哟'],
                                       ['国务院银行业监督管理机构会按照国家法律法规规定的程序处理'],
                                       ['我说点什么好呢？念一个绕口令吧。八了百了标了兵了奔了北了坡，炮了兵了并了排了北了边了跑'],
                                       'test'],
                             title='Empathy-TTS',
                             description='Note: This space is running on CPU, inference times will be higher.')
    interface.launch(server_name='0.0.0.0')
