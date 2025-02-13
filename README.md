# Framework vLLM

Build `Docker Image`
```
docker build -t <docker_image_name>:<tag> -f docker/Dockerfile .
```

Run `Docker Container`
```
docker run \
    --gpus all \
    -it \
    -v .:/workspace \
    <docker_image_name>:<tag>
```

Run script inside `Docker Container` terminal

```
cd workspace
python run_predict_test.py
```

    ВНИМАНИЕ!
    Файлы `models.py` и `run_predict.py` на данный момент не протестированы.
