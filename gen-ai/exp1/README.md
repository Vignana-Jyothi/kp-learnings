# Objectives
Run a basic chatgpt

## Inputs
  Text files  : input.txt

## Output
  Endpoint that gives answers


# Setup

### Python Env setup
``
cd kpenvs/
python3 -m venv GenAI
source ~/kpenvs/GenAI/bin/activate
```

### Install required packages

```
pip install --upgrade pip
pip install -r requirements.txt

```
This installs 
collected packages: mpmath, urllib3, tqdm, sympy, safetensors, regex, psutil, packaging, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, numpy, networkx, fsspec, filelock, charset-normalizer, triton, requests, nvidia-cusparse-cu12, nvidia-cudnn-cu12, nvidia-cusolver-cu12, huggingface-hub, torch, tokenizers, transformers, accelerate

### Download contents
Download text file and save it as : input.txt
MOVE part of the file as : validation.txt


# Run the code
```
python3 model_trainer.py
```

Observe that it generates lot of files
cached_lm_GPT2Tokenizer_128_input.txt
cached_lm_GPT2Tokenizer_128_input.txt.lock
cached_lm_GPT2Tokenizer_128_validation.txt 
cached_lm_GPT2Tokenizer_128_validation.txt.lock
gpt2-finetuned                             
results  

The training process will generate two main directories: `gpt2-finetuned` and `results`. Here’s a detailed explanation of what these directories typically contain:

### Summary of Files

- **`gpt2-finetuned` Directory**:
  - Contains the fine-tuned model ready for deployment.
  - Key files: `config.json`, `pytorch_model.bin`, `tokenizer_config.json`, `vocab.json`, `merges.txt`.

- **`results` Directory**:
  - Contains training progress, checkpoints, and logs.
  - Key files: `checkpoint-*` directories, `training_args.bin`, `all_results.json`.

These files and directories are essential for both using the fine-tuned model and for continuing training or debugging if needed. You can load the model from the `gpt2-finetuned` directory for inference, and you can resume training from any checkpoint in the `results` directory.

A checkpoint is a saved state of the model at a particular point during the training process. Checkpoints are used to preserve the progress of the model so that training can be resumed from that point if needed. They are especially useful for long training processes


# Run 

Run the Server
```
uvicorn chat:app --reload
```

View the Results

```
curl -X POST "http://127.0.0.1:8000/chat" -H "Content-Type: application/json" -d '{"question": "Different types of generative AI models"}'
```