# RWKV

source: https://github.com/BlinkDL/RWKV-LM

preprint: https://arxiv.org/abs/2305.13048

## Introduction

RWKV: Parallelizable RNN with Transformer-level LLM Performance. (pronounced as "RwaKuv", from 4 major params: **R**ecptance **W**eighted **K**ey **V**alue)

>   RWKV is an RNN with transformer-level LLM performance. It can be directly trained like a GPT (parallelizable). So it's combining the best of RNN and transformer - great performance, fast inference, saves VRAM, fast training, "infinite" ctx_len, and free sentence embedding.
>

Genre: Model - Language Model - Algorithm

Download RWKV-4 0.1/0.4/1.5/3/7/14B weights: https://huggingface.co/BlinkDL (Pretrained)

## Core concepts

- RNN = Recurrent Neural Networks （循环神经网络）
>	Transformers have revolutionized almost all natural language processing (NLP) tasks but suffer from memory and computational complexity that scales quadratically with sequence length. In contrast, recurrent neural networks (RNNs) exhibit linear scaling in memory and computational requirements but struggle to match the same performance as Transformers due to limitations in parallelization and scalability. We propose a novel model architecture, Receptance Weighted Key Value (RWKV), that combines the efficient parallelizable training of Transformers with the efficient inference of RNNs. Our approach leverages a linear attention mechanism and allows us to formulate the model as either a Transformer or an RNN, which parallelizes computations during training and maintains constant computational and memory complexity during inference, leading to the first non-transformer architecture to be scaled to tens of billions of parameters. Our experiments reveal that RWKV performs on par with similarly sized Transformers, suggesting that future work can leverage this architecture to create more efficient models. This work presents a significant step towards reconciling the trade-offs between computational efficiency and model performance in sequence processing tasks.
>

## Practical Applications

### Sample App

Raven 14B (finetuned on Alpaca+ShareGPT+...) Demo: https://huggingface.co/spaces/BlinkDL/ChatRWKV-gradio

Raven 7B (finetuned on Alpaca+ShareGPT+...) Demo: https://huggingface.co/spaces/BlinkDL/Raven-RWKV-7B

ChatRWKV: with "stream" and "split" strategies and INT8. 3G VRAM is enough to run RWKV 14B :) https://github.com/BlinkDL/ChatRWKV

https://github.com/josStorer/RWKV-Runner cool GUI

More RWKV projects: https://github.com/search?o=desc&q=rwkv&s=updated&type=Repositories

### Inference (give a pretrained model context to get output)

https://github.com/saharNooby/rwkv.cpp fast int4 int8 fp16 fp32 CPU inference using ggml

https://github.com/harrisonvanderbyl/rwkv-cpp-cuda fast windows/linux & cuda/rocm/vulkan GPU inference (no need for python & pytorch)

https://github.com/Blealtan/RWKV-LM-LoRA **LoRA fine-tuning**

### Local Deployment，Training，Usage Test Result

TBD
