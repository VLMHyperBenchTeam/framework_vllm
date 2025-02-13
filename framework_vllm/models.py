from typing import Any, List

from PIL import Image
from vllm import LLM, SamplingParams
from model_interface.model_interface import ModelInterface


class Qwen2VL2BInstructModel():
    def __init__(self, model_name: str, system_prompt: str, cache_dir: str):
        self.model_name = model_name
        self.system_prompt = system_prompt #TODO how add system prompt
        self.cache_dir = cache_dir
        self.model = LLM(
                        model=self.model_name,
                        max_model_len=4096,
                        max_num_seqs=1,
                        gpu_memory_utilization=0.7,
                        swap_space=2,
                        cache_dir=self.cache_dir,
                    )   
        
    @staticmethod
    def get_prompt(question: str):
        prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                    f"{question}<|im_end|>\n"
                    "<|im_start|>assistant\n")
        return prompt


class FrameworkVllmInterface(ModelInterface):    

    def __init__(
        self,
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        system_prompt="",
        cache_dir="model_cache",
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.cache_dir = cache_dir

        self.model_example_map = {
            "Qwen/Qwen2-VL-2B-Instruct": Qwen2VL2BInstructModel
        }

        if model_name not in self.model_example_map.keys():
            raise ValueError(f"Model type {model_name} is not supported.")

        self.model = self.model_example_map[model_name](self.model_name,
                                                        self.system_prompt,
                                                        self.cache_dir
                                                    )

        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=None,
            stop_token_ids=None
        )
    

    def predict_on_image(self, image, question):        
        
        image = Image.open(image).convert("RGB")
        inputs = {
            "prompt": Qwen2VL2BInstructModel.get_prompt(question),
            "multi_modal_data": {
                "image": image
            },
        }

        outputs = self.model.generate(inputs, sampling_params=self.sampling_params)

        return outputs[0]


    def predict_on_images(self, images: List[Any], question: str) -> str:
        pass

