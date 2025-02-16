# Skyresh

This repo contains the code for both containerization and deployment of a [GLiNER model](https://github.com/urchade/GLiNER) using [OpenVINO](https://docs.openvino.ai/), it is part of a larger personal project. 

**IMPORTANT**: Full compatibility is not guaranteed for different span modes such as `token_level`.


#### Project Status: Active [WIP]

### Technologies
* Language: Python+
* Dependencies: PIP
* Tokenization Framework: Huggingface
* Inference Engine: OpenVINO
* Hardware: CPU
* Model Serving Strategy: FastAPI/Uvicorn
* Container: Docker
* Target Platform: Kubernetes, Docker-Compose, Docker
* Model Repository: MinIO with Datashim

## 🛠 Getting Started

The application exposes both a gradio front-end and a REST-API. You can just pull the image to start working with it:

```bash
docker run --rm --name gliner-vino -p 8080:8080 -p 7860:7860 -it juampahc/skyresh:latest
```

The container does not contain any model, you need to specify the configuration by either:

- using environment variables
- providing an .env file that should be mounted in directory `skyresh/.env`

The application will load default configuration when not provided. Please note that variables present in configuration file will override those passed by environment variables.

For more info about the endpoints check the swagger documentation exposed in `http://localhost:8080/docs`

The gradio front-end allows changing the model by using a different one from huggingface such as `urchade/gliner_multi-v2.1`.

If downloading is not an option for you, you can always mount a repository of models and treat model ID  as model path.

When using this model in production, you need to pass the header 'access_token' (nginx for example). Default API-KEY is `helloworld`.

Examples coming soon!

...


## 👨‍💻 Model Authors
The original model authors are:
* [Urchade Zaratiana](https://huggingface.co/urchade)
* Nadi Tomeh
* Pierre Holat
* Thierry Charnois

Original repo: [https://github.com/urchade/GLiNER](https://github.com/urchade/GLiNER)

## 📚 Citation

This work is based on the following work:

```bibtex
@inproceedings{zaratiana-etal-2024-gliner,
    title = "{GL}i{NER}: Generalist Model for Named Entity Recognition using Bidirectional Transformer",
    author = "Zaratiana, Urchade  and
      Tomeh, Nadi  and
      Holat, Pierre  and
      Charnois, Thierry",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.300",
    doi = "10.18653/v1/2024.naacl-long.300",
    pages = "5364--5376",
    abstract = "Named Entity Recognition (NER) is essential in various Natural Language Processing (NLP) applications. Traditional NER models are effective but limited to a set of predefined entity types. In contrast, Large Language Models (LLMs) can extract arbitrary entities through natural language instructions, offering greater flexibility. However, their size and cost, particularly for those accessed via APIs like ChatGPT, make them impractical in resource-limited scenarios. In this paper, we introduce a compact NER model trained to identify any type of entity. Leveraging a bidirectional transformer encoder, our model, GLiNER, facilitates parallel entity extraction, an advantage over the slow sequential token generation of LLMs. Through comprehensive testing, GLiNER demonstrate strong performance, outperforming both ChatGPT and fine-tuned LLMs in zero-shot evaluations on various NER benchmarks.",
}
```
