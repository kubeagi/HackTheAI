# RWKV 模型

源代码: https://github.com/BlinkDL/RWKV-LM

论文预印本: https://arxiv.org/abs/2305.13048

## 介绍

>   RWKV 是基于循环神经网络的大语言模型算法，其性能与类 transformer 的大语言模型（如 GPT）相媲美，亦可实现并行训练，结合了两类模型的特点：表现优异、快速训练推理、节约显存、“无限”上下文长度、自由调整语句 embedding（特征向量）。
>

分类：模型 - 语言模型 - 算法

中文介绍：https://zhuanlan.zhihu.com/p/603840957

## 特点

- 模型、算法、数据集完全开源
- 推理所需算力小，7B 参数模型仅需约 3GB 显存即可在本地运行
- 基于本地推理的特性，可通过插件实现类似 Stable Diffusion 的精调（fine-tuning），满足个性化自定义需求（参见下列应用）
- 	作者的目标：“做语言模型的 Stable Diffusion”

## 实际应用

### ChatGPT 类应用

在线演示 demo（14B 参数版）：https://huggingface.co/spaces/BlinkDL/ChatRWKV-gradio

在线演示 demo（7B 参数版）: https://huggingface.co/spaces/BlinkDL/Raven-RWKV-7B

ChatRWKV 源代码：https://github.com/BlinkDL/ChatRWKV
中文教程-聊天模型：https://zhuanlan.zhihu.com/p/618011122
中文教程-小说续写模型：https://zhuanlan.zhihu.com/p/609154637
小说模型在线 demo：https://modelscope.cn/studios/BlinkDL/RWKV-CHN/summary
精调版小说模型在线 demo： https://modelscope.cn/studios/BlinkDL/RWKV-CHN-2/summary
中文教程-精调模型：https://zhuanlan.zhihu.com/p/616351661

下载 RWKV-4 预训练权重值: https://huggingface.co/BlinkDL （参数量 0.1/0.4/1.5/3/7/14B）

GUI 界面 https://github.com/josStorer/RWKV-Runner 

其他 RWKV 项目: https://github.com/search?o=desc&q=rwkv&s=updated&type=Repositories
（常见用例有小说续写、聊天机器人等）

### 本地推理

CPU 推理： https://github.com/saharNooby/rwkv.cpp 

GPU 推理： https://github.com/harrisonvanderbyl/rwkv-cpp-cuda 

LoRA 微调：https://github.com/Blealtan/RWKV-LM-LoRA 

### 测试记录

TBD
