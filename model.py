from transformers import AutoModelForCausalLM

class AutoModelForCausalLMBestOfN(AutoModelForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.score

    def 