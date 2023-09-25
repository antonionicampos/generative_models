# Generative Models

## Run on Docker

Docker image to run the code from this repository at the following address.

https://hub.docker.com/r/antonionicampos/tf-generative-models-lps

### Build Image
```docker
docker build --no-cache -t antonionicampos/tf-generative-models-lps:0.0.1 .
```

### Run Container

```docker
docker run --rm -v ./main/images:/generative_models/main/images -it antonionicampos/tf-generative-models-lps:0.0.1 bash
```

- For GPU

```docker
docker run --rm --gpus all -v ./main/images:/generative_models/main/images -it antonionicampos/tf-generative-models-lps:0.0.1 bash
```

### Push Image

To upload a new version of the image, change the `<tagname>` placeholder to an appropriate value.

```docker
docker push antonionicampos/tf-generative-models-lps:<tagname>
```