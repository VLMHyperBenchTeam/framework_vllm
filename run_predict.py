from vllm import LLM, SamplingParams
from PIL import Image

# Qwen2-VL
def run_qwen2_vl(model_name: str, question: str):   

    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=1,
        gpu_memory_utilization=0.7
    )

    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n")
    stop_token_ids = None
    return llm, prompt, stop_token_ids



# Загрузка изображения
def load_image(image_path: str):
    """
    Загружает изображение с помощью PIL.
    """
    return Image.open(image_path).convert("RGB")


if __name__ == "__main__":   

    model_example_map = {
    "qwen2_vl": run_qwen2_vl}

    model_family = "qwen2_vl"
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    image_path = "./images/1.jpg"
    image = load_image(image_path)
    user_prompt  = "What is the content of this image?"

    if model_family not in model_example_map:
        raise ValueError(f"Model type {model_family} is not supported.")

    llm, prompt, stop_token_ids = model_example_map[model_family](model_name, user_prompt)

    sampling_params = SamplingParams(temperature=0,
                                     max_tokens=None,
                                     stop_token_ids=stop_token_ids)
    
    inputs = {
        "prompt": prompt,
        "multi_modal_data": {
            "image": image
        },
    }

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
