import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

# 根据 拼音 构建 音素序列
import re
def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


lexicon = read_lexicon('lexicon/pinyin-lexicon-r.txt')

wave_path = '/home/admin/yuanxin/3.tempData/vits/baker_prosody'

# 划分 训练集、验证集及测试集；并清洗文本，将汉字转换成IPA，同时保留 $ 和 %
wave_filenames = os.listdir(wave_path)
wave_filenames = set([i.split('.')[0] for i in wave_filenames])

file_names_texts = {}
with open('filelists/baker.txt', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i % 2 == 0:
            file = line.split('\t')[0]
            text = line.split('\t')[1].strip('\n')
        if i % 2 == 1:
            phone = line.strip('\n').strip('\t')
            file_names_texts[file + '_t'] = text
            file_names_texts[file + '_p'] = phone


phones_all = []
phone_prosody_all = []
wave_filenames_all = []
for wave_filename in wave_filenames:
    pinyins = file_names_texts[wave_filename + '_p']
    pinyins = pinyins.split(' ')
    text = file_names_texts[wave_filename + '_t']
    text = re.sub(r'[\W]', '', text)
    text = text.replace('1', '#1')
    text = text.replace('2', '#2')
    text = text.replace('3', '#3')
    text = text.replace('4', '#4')
    # 韵律
    prosody = list()
    for j, t in enumerate(text):
        if t in ['#', '1', '2', '3', '4']:
            continue
        else:
            if text[j + 1] == '#' and text[j + 2] == '1':
                prosody.append('B1')
            elif text[j + 1] == '#' and text[j + 2] == '2':
                prosody.append('B2')
            elif text[j + 1] == '#' and text[j + 2] == '3':
                prosody.append('B3')
            else:
                prosody.append('I')

    if len(prosody) != len(pinyins):
        if "儿" in text:
            text = text.replace('儿', '')
            # 重新计算韵律
            prosody = list()
            for j, t in enumerate(text):
                if t in ['#', '1', '2', '3', '4']:
                    continue
                else:
                    if text[j + 1] == '#' and text[j + 2] == '1':
                        prosody.append('B1')
                    elif text[j + 1] == '#' and text[j + 2] == '2':
                        prosody.append('B2')
                    elif text[j + 1] == '#' and text[j + 2] == '3':
                        prosody.append('B3')
                    else:
                        prosody.append('I')
            if len(prosody) != len(pinyins):
                print(wave_filename)
                continue
        else:
            raise OSError('end')

    phones = []
    phone_prosody = []
    for i, p in enumerate(pinyins):
        if p in lexicon:
            phones += lexicon[p]
            for _ in lexicon[p]:
                phone_prosody.append(prosody[i])
        else:
            raise OSError('end')
    phones_all.append(phones)
    phone_prosody_all.append(phone_prosody)
    wave_filenames_all.append(wave_filename)


with open('filelists/mandarin_train.txt', 'w', encoding='utf-8') as f:
    with open('filelists/mandarin_val.txt', 'w', encoding='utf-8') as f1:
        with open('filelists/mandarin_test.txt', 'w', encoding='utf-8') as f2:
            count = 0
            for p, pro, wave_filename in tqdm(zip(phones_all, phone_prosody_all, wave_filenames_all)):
                if count < 512:
                    f1.write("DUMMY2/" + wave_filename + '.wav|' + ' '.join(p) + '|' + ' '.join(pro) + '\n')
                    count += 1
                elif count < 1024:
                    f2.write("DUMMY2/" + wave_filename + '.wav|' + ' '.join(p) + '|' + ' '.join(pro) + '\n')
                    count += 1
                else:
                    f.write("DUMMY2/" + wave_filename + '.wav|' + ' '.join(p) + '|' + ' '.join(pro) + '\n')
                    count += 1
