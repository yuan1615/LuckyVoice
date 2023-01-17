import struct
import re
import os
import torch
import commons
from text import text_to_sequence, prosody_to_sequence


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def get_prosody(text, hps):
    text_norm = prosody_to_sequence(text)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

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


def g2p_mandarin(frontend, text):
    result = frontend.gen_tacotron_symbols(text)
    texts = [s for s in result.splitlines() if s != '']
    pinyin_list = []
    prosody_list = []
    for line in texts:
        pinyin = []
        line = line.strip().split('\t')[1]
        lfeat_symbol = line.strip().split(' ')
        for this_lfeat_symbol in lfeat_symbol:
            this_lfeat_symbol = this_lfeat_symbol.strip('{').strip('}').split(
                '$')
            if this_lfeat_symbol[2] == 's_begin':
                if this_lfeat_symbol[0].split('_')[0] == 'xx':
                    pinyin.append('x')
                else:
                    pinyin.append(this_lfeat_symbol[0].split('_')[0])
            else:
                if this_lfeat_symbol[0].split('_')[0] == 'ih':
                    pinyin.append('iii' + this_lfeat_symbol[1][-1])
                elif this_lfeat_symbol[0].split('_')[0] in ['e', 'an', 'a', 'ang', 'ao', 'ou', 'ong'] and pinyin[-1] == 'y':
                    pinyin.append('i' + this_lfeat_symbol[0].split('_')[0] + this_lfeat_symbol[1][-1])
                else:
                    if this_lfeat_symbol[1] == 'tone_none':
                        pinyin.append(this_lfeat_symbol[0])
                    else:
                        pinyin.append(this_lfeat_symbol[0].split('_')[0] + this_lfeat_symbol[1][-1])
        prosody = ['I' for _ in pinyin]
        for ii, p in enumerate(pinyin):
            if p == '#1':
                prosody[ii - 2:ii] = ['B1', 'B1']
            if p == "#2":
                prosody[ii - 2:ii] = ['B2', 'B2']
            if p == "#3" or p == '#4':
                prosody[ii - 2:ii] = ['B3', 'B3']
        ind = []
        for ii, p in enumerate(pinyin):
            if p in ['ge', 'ga', 'go', '#1', '#2', '#3', '#4']:
                ind.append(ii)
        for a in ind[::-1]:
            pinyin.pop(a)
            prosody.pop(a)
        pinyin = ' '.join(pinyin)
        prosody = ' '.join(prosody)
        pinyin_list.append(pinyin)
        prosody_list.append(prosody)
    return pinyin_list, prosody_list


def create_wav_header(audio_size: int, sampleRate:int, bits:int, channel:int):
    header = b''
    header += b"RIFF"
    header += struct.pack('i', int(audio_size + 44 - 8))
    header += b"WAVEfmt "
    header += b'\x10\x00\x00\x00'
    header += b'\x01\x00'
    header += struct.pack('H', channel)
    header += struct.pack('i', sampleRate)
    header += struct.pack('i', int(sampleRate * bits / 8))
    header += struct.pack('H', int(channel * bits / 8))
    header += struct.pack('H', bits)
    header += b'data'
    header += struct.pack('i', audio_size)
    return header
