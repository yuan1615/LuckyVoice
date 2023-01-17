# LuckyVoice
Inspired by the host competition in 2019, this repository tries to use Zou Yun's voice to build a high-expressive speech synthesis system.
The pinyin of 邹韵 is Zōu yùn, which is a homonym for good luck. 

[HuggingFace🤗 Demo-Baker](https://huggingface.co/spaces/yuan1615/EmpathyTTS) | [HuggingFace🤗 Demo-Lucky | WIP](https://huggingface.co/spaces/EmpathyTTS)


## 1. Data Collection and Processing
### 1.1 Collect related videos of Zou Yun
```
1. Use the 'you-get' tool to download videos in batches, and the video address is in dataprocessing/collectvideos/main.py.
2. Use a format converter to convert video to wav files.
```
### 1.2 Split the audio using the [vad](https://github.com/snakers4/silero-vad) method.
```
python dataprocessing/vad/main.py --pth [downloaded video] --savepth [Save address of split audio]
```

### 1.3 Noise reduction using [speech enhancement model](https://www.modelscope.cn/models/damo/speech_frcrn_ans_cirm_16k/summary).
[pre-trained model](https://drive.google.com/file/d/1T0fm9GA_0PIg8QOchpnHcdG9Kvp_X0ZN/view?usp=sharing)
```
sudo docker build -t se .
sudo docker run -it --rm -v /home/admin/yuanxin:/se se
python dataprocessing/se/main.py
```
### 1.4 Classify audio using a [voiceprint recognition model](https://github.com/wenet-e2e/wespeaker).


### 1.5 Processing text with a [speech recognition](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) and [speech synthesis front-end](https://www.modelscope.cn/models/damo/speech_sambert-hifigan_tts_zhitian_emo_zh-cn_16k/summary)
[speech synthesis front-end](https://drive.google.com/file/d/1jAfnclbgAkUXXKjWgBic2dmdPJECQgzm/view?usp=sharing)

## 2. Baseline Model

### 2.1 [VITS](https://github.com/jaywalnut310/vits) model with prosodic representation
[pretrained_baker.pth](https://drive.google.com/file/d/13IJf70A5UjvTfJBMowVGjXLTpERaYZnV/view?usp=sharing)
```
python model/vits/main.py --text ['你好'] --out [The address to save the file]
```

### 2.2 [DiffSpeech](https://github.com/MoonInTheRiver/DiffSinger) model with prosodic representation


## 3. EmpathyTTS

