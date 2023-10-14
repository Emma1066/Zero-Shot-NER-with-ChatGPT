# Empirical Study of Zero-Shot NER with ChatGPT

This is the github repository for the paper to be appeared at EMNLP 2023: Empirical Study of Zero-Shot NER with ChatGPT.

## Introduction

This work focuses on exploring LLM performance on zero-shot information extraction, with a focus on the ChatGPT and named entity recognition (NER) task.

![](figs/method.jpg)

Inspired by the remarkable reasoning capability of LLM on symbolic and arithmetic reasoning, we adapt the prevalent reasoning methods to NER and propose reasoning strategies tailored for NER:
* We break down the NER task into a series of simpler subproblems by labels and perform a decomposed-question-answering (**Decomposed-QA**) paradigm, where the model extracts entities of only one label at a time.
* We propose syntactic augmentation of two ways: **syntactic prompting**, which encourages the model to first analyze the syntactic structure of the input text itself, then recognize the named entities based on the syntactic structure; **tool augmentation**, which provides the syntactic information generated by a parsing tool to the model.
* We tailor the self-consistency (SC) for NER and propose a **two-stage majority voting strategy**: after sampling multiple responses of the model, we first vote for the most consistent mentions, then the most consistent types.

Please find more details of this work in our paper.

## Usage

### Dependencies
python 3.8, openai 0.27.4, pytorch 2.0.1, pandas, hanlp

### Datasets
We provide processed datasets used in our paper at the [google drive](https:), except [ACE04](https://catalog.ldc.upenn.edu/LDC2005T09), [ACE05](https://catalog.ldc.upenn.edu/LDC2006T06) and [Ontonotes 4](https://catalog.ldc.upenn.edu/LDC2011T03) for copyright reasons. Power plant datasets, PowerPlantFlat and PowerPlantNested, involve ongoing collaborative projects with our partners, and we will release them after the project is completed and the data is desensitized.

Download dataset files, unzip them, and place them in the data folder.

### Run
The shell scripts in folder scripts can be used to quickly run the pipeline of generating prompts, run LLM and computing the metrics of results.

Vanilla
```
```

Decomposed-QA
```
```

The following three mthods are used under Decomposed-QA paradigm.
Syntactic prompting. POS tagger is taken for an example.
```
```

Tool augmentation. POS tagger is taken for an example.
```
```

Tool augmentation plus syntactic prompting. POS tagger is taken for an example.
```
```

The combination of tool augmentation, syntactic promoting and self-consistency. POS tagger is taken for an example.
```
```

### Generate syntactic information
Run the following command to generate syntactic information with [Hanlp](https://github.com/hankcs/HanLP).
```
```

### Other LLMs
Our code is compatible to other LLMs deployed with OpenAI API. We deploy the Llama2 model with OpenAI API using [FastChat](https://github.com/lm-sys/FastChat). The we change the argument --model to the corresponding model name to run Llama2.

You can obtain the access to Llama2 in [Meta official website](https://ai.meta.com/llama/) and [Huggingface](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf).

## Contact

If you have any issues or questions about this repo, feel free to contact tingyuxie@zju.edu.cn.
