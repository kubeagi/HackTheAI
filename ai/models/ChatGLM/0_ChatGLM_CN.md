# ChatGLM

模型：[THUDM/GLM: GLM (General Language Model) (github.com)](https://github.com/THUDM/GLM)

对话模型： [THUDM/ChatGLM2-6B: ChatGLM2-6B: An Open Bilingual Chat LLM | 开源双语对话语言模型 (github.com)](https://github.com/THUDM/ChatGLM2-6B)

微调： [ChatGLM-6B/ptuning/README.md at main · THUDM/ChatGLM-6B · GitHub](https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/README.md)

## 介绍

> ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 General Language Model (GLM) 架构，具有 62 亿参数。结合模型量化技术，用户可以在消费级的显卡上进行本地部署（INT4 量化级别下最低只需 6GB 显存）。 ChatGLM-6B 使用了和 ChatGPT 相似的技术，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持，62 亿参数的 ChatGLM-6B 已经能生成相当符合人类偏好的回答。

>   ChatGLM2-6B 是开源中英双语对话模型 [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) 的第二代版本，在保留了初代模型对话流畅、部署门槛较低等众多优秀特性的基础之上，ChatGLM2-6B 引入了如下新特性：
>
>   1.  **更强大的性能**：基于 ChatGLM 初代模型的开发经验，我们全面升级了 ChatGLM2-6B 的基座模型。ChatGLM2-6B 使用了 [GLM](https://github.com/THUDM/GLM) 的混合目标函数，经过了 1.4T 中英标识符的预训练与人类偏好对齐训练，[评测结果](https://github.com/THUDM/ChatGLM2-6B#评测结果)显示，相比于初代模型，ChatGLM2-6B 在 MMLU（+23%）、CEval（+33%）、GSM8K（+571%） 、BBH（+60%）等数据集上的性能取得了大幅度的提升，在同尺寸开源模型中具有较强的竞争力。
>   2.  **更长的上下文**：基于 [FlashAttention](https://github.com/HazyResearch/flash-attention) 技术，我们将基座模型的上下文长度（Context Length）由 ChatGLM-6B 的 2K 扩展到了 32K，并在对话阶段使用 8K 的上下文长度训练，允许更多轮次的对话。但当前版本的 ChatGLM2-6B 对单轮超长文档的理解能力有限，我们会在后续迭代升级中着重进行优化。
>   3.  **更高效的推理**：基于 [Multi-Query Attention](http://arxiv.org/abs/1911.02150) 技术，ChatGLM2-6B 有更高效的推理速度和更低的显存占用：在官方的模型实现下，推理速度相比初代提升了 42%，INT4 量化下，6G 显存支持的对话长度由 1K 提升到了 8K。
>   4.  **更开放的协议**：ChatGLM2-6B 权重对学术研究**完全开放**，在获得官方的书面许可后，亦**允许商业使用**。如果您发现我们的开源模型对您的业务有用，我们欢迎您对下一代模型 ChatGLM3 研发的捐赠。

## 特点

-   参数开源
-   下游应用生态发展良好，已有多种知识库类应用及功能拓展
-   消费级显卡推理部署（fp16 须 13GB 显存）

## 实践

langchain+ChatGLM 实现本地知识库：

-   源代码： [imClumsyPanda/langchain-ChatGLM: langchain-ChatGLM, local knowledge based ChatGLM with langchain ｜ 基于本地知识库的 ChatGLM 问答 (github.com)](https://github.com/imClumsyPanda/langchain-ChatGLM)

-   介绍：[ChatGLM-6B 结合 langchain 实现本地知识库 QA Bot - Heywhale.com](https://www.heywhale.com/mw/project/643977aa446c45f4592a1e59)


引入网络查询：[THUDM/WebGLM: WebGLM: An Efficient Web-enhanced Question Answering System (KDD 2023) (github.com)](https://github.com/THUDM/WebGLM)

引入图像识别：[THUDM/VisualGLM-6B: Chinese and English multimodal conversational language model | 多模态中英双语对话语言模型 (github.com)](https://github.com/THUDM/VisualGLM-6B)

C++ 轻量化版本：[MegEngine/InferLLM: a lightweight LLM model inference framework (github.com)](https://github.com/MegEngine/InferLLM)

## 测试记录

TBD