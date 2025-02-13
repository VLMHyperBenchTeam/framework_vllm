import os
import subprocess

from model_interface.model_factory import ModelFactory


if __name__ == "__main__":

    cache_directory = "model_cache"

    # Сохраняем модели Qwen2-VL в примонтированную папку
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_directory = os.path.join(script_dir, cache_directory)

    # Имена моделей и семейство моделей
    model_name_1 = "Qwen/Qwen2-VL-2B-Instruct"
    model_family = "vLLM"

    # Инфо о том где взять класс для семейства моделей
    package = "framework_vllm"
    module = "models"
    model_class = "FrameworkVllmInterface"
    model_class_path = f"{package}.{module}:{model_class}"

    # Регистрация модели в фабрике
    ModelFactory.register_model(model_family, model_class_path)

    # создаем модель
    model_init_params = {
        "model_name": model_name_1,
        "system_prompt": "",
        "cache_dir": "model_cache",
        "device_map": "cuda:0",
    }

    model = ModelFactory.get_model(model_family, model_init_params)

    # отвечаем на вопрос о по одной картинке
    image_path = "./images/1.jpg"
    question = "Какой текст на изображении?"
    model_answer = model.predict_on_image(image=image_path, question=question)
    print(model_answer)
    
    subprocess.run(["nvidia-smi"])

    print(model_answer)
