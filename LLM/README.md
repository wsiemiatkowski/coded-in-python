# LLMs

## Overview

This section of the repository contains scripts that interface with locally hosted large language models (LLMs).

### How to run the code
 
Ollama is required to run the scripts.
1. Download [Ollama(https://ollama.com/download)
2. Open terminal/shell
3. Pull the required model corresponding to the script you want to run:
```shell
  ollama pull <model_name>
```
4. Create an ollama server
```shell
  ollama serve
```
Note: make sure that port used for your Ollama server corresponds to the address given in `utils/.env`. 
Port 11434 is standard for this service however it may vary in individual cases when it's already in use on your machine.

### Utils
Utils houses the crucial pieces to the scripts in LLM directory. In `.env` you will find the address and model names 
for this project. `LLMPrompter.py` contains classes that allows inferencing of the Ollama hosted models.
Class `LLMPrompter` can be used for simple chatting with LLM, while LLMPrompterFewShot contains examples
that provide more context for the model to use.