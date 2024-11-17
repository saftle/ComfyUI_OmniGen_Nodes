# This code is used to call some components of OmniGen in a different order
import os
from transformers import AutoTokenizer

from OmniGen import OmniGenProcessor


class OmniGenProcessorWrapper(OmniGenProcessor):
    @classmethod
    def from_pretrained(cls):
        text_tokenizer = AutoTokenizer.from_pretrained(os.path.join(os.path.dirname(__file__), 'tokenizer'))
        return cls(text_tokenizer)
