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

1.   fastchat 运行模型，打包为 openai api server

使用 Fastchat 构建基于 Chat2GLM-6B 的 OpenAI API Server:

```dockerfile
RUN python3 -m fastchat.serve.controller
RUN python3 -m fastchat.serve.model_worker --model-path ~/bjwswang/ai-modesl/chatglm-6b --host 0.0.0.0
RUN python3 -m fastchat.serve.openai_api_server --host 0.0.0.0
```

启动后，使用 curl 输入如下指令测试：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chatglm-6b",
    "messages": [{"role": "user", "content": "Hello! What is your name?"}]
  }'
```

得到回复：

```json
{"id":"chatcmpl-3kUj7X4byD38b4Bp73o84N",
"object":"chat.completion",
"created":1689316550,
"model":"chatglm-6b",
"choices":
	[{"index":0,
	"message":{
		"role":"assistant",
		"content":"Hello! I am ChatGLM2-6B, a language model jointly trained by KEG Lab of Tsinghua University and Zhipu AI Company."
		},
	"finish_reason":"stop"
	}],
"usage":{
	"prompt_tokens":21,
	"total_tokens":56,
	"completion_tokens":35
	}
}
```

2.   设置 host，使用 gradio 应用进行调试：

```python
import gradio as gr
import openai
from gradio.components import Textbox

openai.api_key = "EMPTY"
openai.api_base = "http://fastchat.ai.com/v1"

model = "chatglm-6b"


def GLM_Bot(input):
    if input:
        response = openai.Completion.create(model=model, prompt=input, max_tokens=128)
        result = response.choices[0].text
        return input + result


inputs = Textbox(lines=7, label="请输入补全提示词")
outputs = Textbox(label="GLM 生成结果")

demo = gr.Interface(fn=GLM_Bot, inputs=inputs, outputs=outputs, title="GLM Demo",
                    description="以下是基于 ChatGLM2-6B 模型，在 FastChat 上构建 openai api server 的补全调用演示",
                    theme="Default")

demo.launch(share=True)
```

结果如下：

![api_test_01](..\..\..\images\api_test_01.jpg)

3.   与 Langchain 对接（基于 `..\..\vectorstores\ask.py` ）

Fastchat 与 Langchain 的对接通过在 `model_name` 中伪装成 OpenAI 模型的方法实现：

替换构建过程的第二步为：

```
python3 -m fastchat.serve.model_worker --model-names "gpt-3.5-turbo,text-davinci-003,text-embedding-ada-002" --model-path ~/bjwswang/ai-modesl/chatglm-6b --host 0.0.0.0
```

注意，上述命令使文本处理、特征化及 LLM 推理都使用同一模型。实际应用中可同时运行并注册其他特征化模型，如 `text2vec` 等。

测试文本为一篇拜登总统的演讲稿：（`https://raw.githubusercontent.com/hwchase17/langchain/v0.0.200/docs/modules/state_of_the_union.txt`）

通过修改过的 `ask.py` 处理该演讲稿，并询问问题：

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from typing import List
from langchain.schema import Document
import os

# 替换 Openai API 为本地部署的伪装实例
os.environ['OPENAI_API_KEY'] = "EMPTY"
os.environ['OPENAI_API_BASE'] = "http://fastchat.ai.com/v1"


class Genie:

    def __init__(self, file_path: str):
        # 数据文件
        self.file_path = file_path
        self.loader = TextLoader(self.file_path, encoding='utf-8')
        # 通过loader变成[Document]
        self.documents = self.loader.load()
        # 将[Document]切割成更小的
        self.texts = self.text_split(self.documents)
        self.vectordb = self.embeddings(self.texts)
        self.genie = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff",
                                                 retriever=self.vectordb.as_retriever())

    @staticmethod
    def text_split(documents: TextLoader):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        return texts

    @staticmethod
    def embeddings(texts: List[Document]):
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(texts, embeddings)
        return vectordb

    def ask(self, query: str):
        return self.genie.run(query)


if __name__ == "__main__":
    genie = Genie("./example.txt")
    print(genie.ask("Who are mentioned in the speech?"))

```

ChatGLM2-6B 回复（运行耗时约 20 秒）：

```
The speaker mentions the American people, the State of the Union, and various government officials and employees.
```

评价：相对准确，但回答比较宽泛。

问题 2：

```
How many projects were announced?
```

ChatGLM2-6B 回复：

```
The first project is cutting the cost in half for most families and helping parents return to work. The second project is home and long-term care, more affordable housing, and Pre-K for every 3- and 4-year-old. The third project is ending the shutdown of schools and businesses. So the final answer is three.
```

评价：出现了【幻觉】，原文中并未原样提及上述几个项目。