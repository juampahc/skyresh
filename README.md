# Skyresh

This repo contains the code for both containerization and deployment of a [GLiNER model](https://github.com/urchade/GLiNER) using [OpenVINO](https://docs.openvino.ai/), it is part of a larger personal project. 



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



### Docker

The image of this project is already 
1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data is being kept [here](Repo folder containing raw data) within this repo.


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
