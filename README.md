# Generative Models

## Run on Docker

```docker
docker build --no-cache -t antonionicampos/tensorflow-lps:0.0.1 .
docker run --rm --gpus all -v ./main/images:/generative_models/main/images -it antonionicampos/tensorflow-lps:0.0.1 bash
```