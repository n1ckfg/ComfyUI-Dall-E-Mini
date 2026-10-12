"""Prompt normalisation and tokenisation (port of ``dalle_mini/model/text.py`` and ``processor.py``).

The models were trained on normalised captions, so prompts go through the same pipeline.
"""

import html
import math
import random
import re
import unicodedata
import warnings
import zlib

import torch

# based on wiki word occurrence
_PERSON_TOKENS = [("a person", 282265), ("someone", 121194), ("somebody", 12219)]
_TEMP = "xtokx"  # avoid repeating chars


class HashtagProcessor:
    """Splits ``#sunsetlake`` into ``sunset lake`` with a word-frequency list (adapted from wordninja)."""

    def __init__(self, word_frequency_file):
        with open(word_frequency_file, "r", encoding="utf8") as f:
            words = (line.split()[0] for line in f.read().splitlines())
            self._word_cost = {str(word): math.log(float(i + 1)) for i, word in enumerate(words)}
        self._max_word = max(len(word) for word in self._word_cost)
        self._split_re = re.compile("[^a-zA-Z0-9']+")

    def __call__(self, s):
        parts = [self._split(x) for x in self._split_re.split(s)]
        return " ".join(item for sublist in parts for item in sublist)

    def _split(self, s):
        # Best match for the first i characters, given the cost of the first i-1: (match_cost, match_length)
        def best_match(i):
            candidates = enumerate(reversed(cost[max(0, i - self._max_word):i]))
            return min((c + self._word_cost.get(s[i - k - 1:i].lower(), math.inf), k + 1) for k, c in candidates)

        cost = [0]
        for i in range(1, len(s) + 1):
            c, _ = best_match(i)
            cost.append(c)

        # backtrack to recover the minimal-cost split
        out = []
        i = len(s)
        while i > 0:
            _, k = best_match(i)
            new_token = True
            if not s[i - k:i] == "'" and len(out) > 0:
                # re-attach split 's and split digits
                if out[-1] == "'s" or (s[i - 1].isdigit() and out[-1][0].isdigit()):
                    out[-1] = s[i - k:i] + out[-1]
                    new_token = False
            if new_token:
                out.append(s[i - k:i])
            i -= k
        return reversed(out)


def _replace_person_token(t):
    "Used for CC12M"
    t = re.sub(r"<person>([,\s]*(and)*[,\s]*<person>)+", " people ", t)
    if "<person>" in t:
        # upstream picks at random; seed from the text so a prompt always normalises the same way
        rng = random.Random(zlib.crc32(t.encode("utf8")))
        while "<person>" in t:
            t = t.replace("<person>", f" {rng.choices(*tuple(zip(*_PERSON_TOKENS)))[0]} ", 1)
    return t


def _remove_comma_numbers(t):
    for _ in range(2):
        t = re.sub(r"(\d),(\d{3})", r"\1\2", t)
    return t


def _handle_special_chars(t):
    # replace "-" with a space when between words without space
    t = re.sub(r"(\w)-(\w)", r"\1 \2", t)
    # always add space around some characters
    return re.sub(r"([%&\/$*])", r" \1 ", t)


def _remove_first_last_commas(t):
    t = t.strip()
    t = t[:-1] if t and t[-1] == "," else t
    t = t[1:] if t and t[0] == "," else t
    return t.strip()


def _remove_wiki_ref(t):
    t = re.sub(r"\A\s*\[\d+\]", "", t)
    return re.sub(r"\[\d+\]\s*\Z", "", t)


def _optional(module_name, purpose):
    try:
        return __import__(module_name)
    except ImportError:
        warnings.warn(f"DALL·E Mini: python package '{module_name}' is not installed, {purpose}. "
                      f"Install it with: pip install {module_name}")
        return None


class TextNormalizer:
    """``word_frequency_file`` is a path, or a callable returning one (it is a 34 MB download that only
    prompts containing a #hashtag need, so it is resolved on first use)."""

    def __init__(self, word_frequency_file=None):
        self._word_frequency_file = word_frequency_file
        self._hashtag_processor = None
        self._libs = None

    def _load_libs(self):
        if self._libs is None:
            self._libs = (
                _optional("ftfy", "broken unicode in prompts will not be repaired"),
                _optional("emoji", "emoji in prompts will be dropped instead of described"),
                _optional("unidecode", "non-latin characters in prompts will be dropped instead of transliterated"),
            )
        return self._libs

    def _expand_hashtag(self, word):
        if self._hashtag_processor is None:
            source = self._word_frequency_file
            try:
                path = source() if callable(source) else source
                self._hashtag_processor = HashtagProcessor(path) if path else False
            except Exception as e:
                warnings.warn(f"DALL·E Mini: could not load the word list used to split hashtags ({e})")
                self._hashtag_processor = False
        return self._hashtag_processor(word) if self._hashtag_processor else word

    def __call__(self, t):
        ftfy, emoji, unidecode = self._load_libs()
        # fix some characters
        if ftfy is not None:
            t = ftfy.fix_text(t)
        # fix html (twice, as in OpenAI CLIP: web captions are often double-escaped)
        t = html.unescape(html.unescape(t))
        # decode emojis (would be removed by unidecode)
        if emoji is not None:
            t = emoji.demojize(t)
        # decode and simplify text
        if unidecode is not None:
            t = unidecode.unidecode(t)
        else:
            t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
        t = t.lower()
        # replace <PERSON> (for CC12M)
        t = _replace_person_token(t)
        # remove wiki reference (for WIT)
        t = _remove_wiki_ref(t)
        # remove html tags
        t = re.sub(r"<[^<]+?>", " ", t)
        # remove urls
        t = re.sub(r"http\S+", "", t)
        # remove commas in numbers
        t = _remove_comma_numbers(t)
        # protect dots in numbers, apostrophes ('s, 't, 'd, 'm, 'll, 're, 've) and dates - part 1
        t = re.sub(r"(\w)\.(\w)", rf"\1{_TEMP}dot{_TEMP}\2", t)
        t = re.sub(r"'(?=([stdm]|(ll)|(re)|(ve)|(ll))\b)", rf"{_TEMP}quote{_TEMP}", t)
        t = re.sub(r"(\d)/(\d)", rf"\1{_TEMP}slash{_TEMP}\2", t)
        # handle special characters
        t = _handle_special_chars(t)
        # remove # and try to split words
        t = re.sub(r"#(\w+)", lambda m: self._expand_hashtag(m.group(1)), t)
        # ignore useless characters
        t = re.sub(r"[_#\\]", " ", t)
        # simplify quotes
        t = re.sub(r"""['"`]""", ' " ', t)
        # all punctuation becomes commas
        t = re.sub(r"[()[\].,|:;?!=+~\-\/{}]", ",", t)
        # protect dots in numbers, apostrophes and dates - part 2
        t = t.replace(f"{_TEMP}dot{_TEMP}", ".")
        t = t.replace(f"{_TEMP}quote{_TEMP}", "'")
        t = t.replace(f"{_TEMP}slash{_TEMP}", "/")
        # the same character 4+ times (not 3 because of roman 'VIII') becomes a single instance
        t = re.sub(r"(\D)(\1{3,})", r"\1", t)
        # merge quotes
        t = re.sub(r'(\s*"+\s*)+', ' " ', t)
        # merge commas
        t = re.sub(r"(\s*,+\s*)+", ", ", t)
        # remove extra spaces (including \t and \n)
        t = re.sub(r"\s+", " ", t)
        # remove first and last comma
        t = _remove_first_last_commas(t)
        # always start with a space
        return f" {t}"


class TextProcessor:
    """Prompt -> padded token ids and attention mask, as ``DalleBartProcessor`` does."""

    def __init__(self, tokenizer_file, config, word_frequency_file=None):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(tokenizer_file)
        pad_id = self.tokenizer.token_to_id("<pad>")
        self.tokenizer.enable_truncation(max_length=config.max_text_length)
        self.tokenizer.enable_padding(length=config.max_text_length, pad_id=pad_id, pad_token="<pad>")
        self.unk_id = self.tokenizer.token_to_id("<unk>")
        self.vocab_size = config.encoder_vocab_size
        self.normalizer = TextNormalizer(word_frequency_file) if config.normalize_text else None

    def normalize(self, text):
        # Training blanked captions to "" after normalisation to teach unconditional generation, so an
        # empty prompt must reach the tokenizer as "" rather than as the normaliser's " ".
        if not text.strip():
            return ""
        return self.normalizer(text) if self.normalizer is not None else text

    def __call__(self, text):
        """Returns (input_ids, attention_mask), both int64 of shape [1, max_text_length]."""
        encoding = self.tokenizer.encode(self.normalize(text))
        input_ids = torch.tensor([encoding.ids], dtype=torch.long)
        # the tokenizer knows a few more tokens than the text embedding has rows
        input_ids[input_ids >= self.vocab_size] = self.unk_id
        return input_ids, torch.tensor([encoding.attention_mask], dtype=torch.long)
