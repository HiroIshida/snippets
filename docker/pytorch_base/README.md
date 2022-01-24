```bash
docker build -t pytorch_base .
docker run -it \
    --name pytorch_base_container \
    -it --gpus all \
    pytorch_base:latest
```
