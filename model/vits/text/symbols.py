""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.
'''
_pad        = '_'
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

from text import pinyin, cmudict

_pinyin = ["@" + s for s in pinyin.valid_symbols]
_arpabet = ["@" + s for s in cmudict.valid_symbols]
_punctuation = ["@" + s for s in _punctuation]
_se = ["@'", "@[start]", "@[end]"]

# Export all symbols:
# symbols = [_pad] + _pinyin + _arpabet + _se + _punctuation
symbols = [_pad] + _pinyin

# Special symbol ids
# SPACE_ID = symbols.index(" ")
