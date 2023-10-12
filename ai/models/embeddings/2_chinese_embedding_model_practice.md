# 中文 Embedding 模型实践

## 常见模型

-   text2vec 系列

    -    [shibing624/text2vec: text2vec, text to vector. 文本向量表征工具，把文本转化为向量矩阵，实现了Word2Vec、RankBM25、Sentence-BERT、CoSENT等文本表征、文本相似度计算模型，开箱即用。 (github.com)](https://github.com/shibing624/text2vec)

    >   -   `shibing624/text2vec-base-chinese`模型，是用CoSENT方法训练，基于`hfl/chinese-macbert-base`在中文STS-B数据训练得到，并在中文STS-B测试集评估达到较好效果，运行[examples/training_sup_text_matching_model.py](https://github.com/shibing624/text2vec/blob/master/examples/training_sup_text_matching_model.py)代码可训练模型，模型文件已经上传HF model hub，中文通用语义匹配任务推荐使用
    >   -   `shibing624/text2vec-base-chinese-sentence`模型，是用CoSENT方法训练，基于`nghuyong/ernie-3.0-base-zh`用人工挑选后的中文STS数据集[shibing624/nli-zh-all/text2vec-base-chinese-sentence-dataset](https://huggingface.co/datasets/shibing624/nli-zh-all/tree/main/text2vec-base-chinese-sentence-dataset)训练得到，并在中文各NLI测试集评估达到较好效果，运行[examples/training_sup_text_matching_model_jsonl_data.py](https://github.com/shibing624/text2vec/blob/master/examples/training_sup_text_matching_model_jsonl_data.py)代码可训练模型，模型文件已经上传HF model hub，中文s2s(句子vs句子)语义匹配任务推荐使用
    >   -   `shibing624/text2vec-base-chinese-paraphrase`模型，是用CoSENT方法训练，基于`nghuyong/ernie-3.0-base-zh`用人工挑选后的中文STS数据集[shibing624/nli-zh-all/text2vec-base-chinese-paraphrase-dataset](https://huggingface.co/datasets/shibing624/nli-zh-all/tree/main/text2vec-base-chinese-paraphrase-dataset)，数据集相对于[shibing624/nli-zh-all/text2vec-base-chinese-sentence-dataset](https://huggingface.co/datasets/shibing624/nli-zh-all/tree/main/text2vec-base-chinese-sentence-dataset)加入了s2p(sentence to paraphrase)数据，强化了其长文本的表征能力，并在中文各NLI测试集评估达到SOTA，运行[examples/training_sup_text_matching_model_jsonl_data.py](https://github.com/shibing624/text2vec/blob/master/examples/training_sup_text_matching_model_jsonl_data.py)代码可训练模型，模型文件已经上传HF model hub，中文s2p(句子vs段落)语义匹配任务推荐使用
    >   -   `shibing624/text2vec-base-multilingual`模型，是用CoSENT方法训练，基于`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`用人工挑选后的多语言STS数据集[shibing624/nli-zh-all/text2vec-base-multilingual-dataset](https://huggingface.co/datasets/shibing624/nli-zh-all/tree/main/text2vec-base-multilingual-dataset)训练得到，并在中英文测试集评估相对于原模型效果有提升，运行[examples/training_sup_text_matching_model_jsonl_data.py](https://github.com/shibing624/text2vec/blob/master/examples/training_sup_text_matching_model_jsonl_data.py)代码可训练模型，模型文件已经上传HF model hub，多语言语义匹配任务推荐使用
    >   -   `shibing624/text2vec-bge-large-chinese`模型，是用CoSENT方法训练，基于`BAAI/bge-large-zh-noinstruct`用人工挑选后的中文STS数据集[shibing624/nli-zh-all/text2vec-base-chinese-paraphrase-dataset](https://huggingface.co/datasets/shibing624/nli-zh-all/tree/main/text2vec-base-chinese-paraphrase-dataset)训练得到，并在中文测试集评估相对于原模型效果有提升，在短文本区分度上提升明显，运行[examples/training_sup_text_matching_model_jsonl_data.py](https://github.com/shibing624/text2vec/blob/master/examples/training_sup_text_matching_model_jsonl_data.py)代码可训练模型，模型文件已经上传HF model hub，中文s2s(句子vs句子)语义匹配任务推荐使用

-   moka-ai/m3e-base

    -   模型：[moka-ai/m3e-base · Hugging Face](https://huggingface.co/moka-ai/m3e-base)

-   百度文心/ernie-3.0

    -   [PaddlePaddle/ERNIE: Official implementations for various pre-training models of ERNIE-family, covering topics of Language Understanding & Generation, Multimodal Understanding & Generation, and beyond. (github.com)](https://github.com/PaddlePaddle/ERNIE)
    -   模型：[nghuyong/ernie-3.0-base-zh · Hugging Face](https://huggingface.co/nghuyong/ernie-3.0-base-zh)

-   达摩/coROM

    -   模型：[CoROM文本向量-中文-通用领域-base · 模型库 (modelscope.cn)](https://modelscope.cn/models/damo/nlp_corom_sentence-embedding_chinese-base/summary)

-   商汤/piccolo

    -   模型：[sensenova/piccolo-base-zh · Hugging Face](https://huggingface.co/sensenova/piccolo-base-zh)

-   北京智源/bge

    -   [FlagOpen/FlagEmbedding: Open-source Embedding Models and Ranking Models (github.com)](https://github.com/FlagOpen/FlagEmbedding)
    -   模型：[BAAI/bge-base-zh-v1.5 · Hugging Face](https://huggingface.co/BAAI/bge-base-zh-v1.5)

-   stella（piccolo 改进）

    -   [SmartLi8/stella: text embedding (github.com)](https://github.com/SmartLi8/stella)
    -   模型：[infgrad/stella-base-zh · Hugging Face](https://huggingface.co/infgrad/stella-base-zh)

## 运行参数（Langchain Embedding）

### 通用参数

-   分割方式 Text Splitter
-   分块大小 Chunk Size （以 token 数计算）
-   分块上下文长度 Chunk Overlap（以 token 数计算，用于增加不同片段之间的联系）

### 文档参数

-   PDF
    -   密码 password（`string`，适用于设置了密码的 pdf 文件）
-   CSV
    -   列 columns（`[]string`，筛选需要读取的表格列）

## 模型评估方法

### C-MTEB

-   [FlagEmbedding/C_MTEB at master · FlagOpen/FlagEmbedding (github.com)](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB)

-   （基于 MTEB https://github.com/embeddings-benchmark/mteb）

-   榜单 Leaderboard：[MTEB Leaderboard - a Hugging Face Space by mteb](https://huggingface.co/spaces/mteb/leaderboard)

-   共 6 类任务：

    -   Classification 分类（9 数据集）：根据已有类别，归类输入文本
    -   Clustering 聚类（4 数据集）：直接将输入文本分类
    -   Pair Classification 句子对分类（2 数据集）：判断一对文本是否属于同一类（输出为是/否）
    -   Reranking 重排序（4 数据集）：根据输入，对文本集做相关性重排序
    -   Retrieval 检索（8 数据集）：根据输入，检索文本集中相应文本
    -   Semantic Textual Similarity - STS 文本语义相似度：（8 数据集）检测输入文本的相似度（输出为 0~1 之间的数值）

-   使用方法：

    `pip install -U C_MTEB`

    -   使用 `sentense_transformer` 库：

    ```python
    from mteb import MTEB
    from C_MTEB import *
    from sentence_transformers import SentenceTransformer
    
    # Define the sentence-transformers model name
    model_name = "bert-base-uncased"
    
    model = SentenceTransformer(model_name)
    evaluation = MTEB(task_langs=['zh'])
    results = evaluation.run(model, output_folder=f"zh_results/{model_name}")
    ```

    -   不使用：

    ```python
    class MyModel():
        def encode(self, sentences, batch_size=32, **kwargs):
            """ Returns a list of embeddings for the given sentences.
            Args:
                sentences (`List[str]`): List of sentences to encode
                batch_size (`int`): Batch size for the encoding
    
            Returns:
                `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
            """
            pass
    
    model = MyModel()
    evaluation = MTEB(tasks=["T2Retrival"])
    evaluation.run(model)
    ```

    此处 `encode()` 函数需要根据模型编写并适配。

-   测试资源消耗：以 `text2vec-base-chinese` 模型为例，该模型在 4090 显卡上满载运行，测试全过程耗时约 2 小时。

-   测试结果形式：

    -   检索类（Retrieval）：（关注 `ndcg_at_10` 项）
        -   NDCG = 归一化折损累计增益
        -   MRR = 平均倒数排名
        -   MAP =  Mean Average Precision = 平均精度均值
            -   对 NDCG 和 MRR，关联度越高的结果越能靠前，则得分越高
            -   对 MAP，越能精准地找到更多数据，则得分越高
            -   **上述三项得分数值越高越好**
        -   Recall = 查全率，需要搜的全部数据中**被搜索到**的比例
        -   Precision = 精准率，搜索到的结果中**符合要求**的比例
        -   source：
        -   [推荐系统常用指标（续）MRR、HR、MAP，Fscore - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/433081516)
        -   [NDCG 归一化折损累计增益的动机、讲解、公式、实现 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/474423793)

    ```json
    {
      "dataset_revision": null,
      "dev": {
        "evaluation_time": 145.56,
        "map_at_1": 0.16121,
        "map_at_10": 0.40117,
        "map_at_100": 0.43886,
        "map_at_1000": 0.44131,
        "map_at_3": 0.29409,
        "map_at_5": 0.35094,
        "mrr_at_1": 0.63247,
        "mrr_at_10": 0.69908,
        "mrr_at_100": 0.703,
        "mrr_at_1000": 0.7032,
        "mrr_at_3": 0.68416,
        "mrr_at_5": 0.69301,
        "ndcg_at_1": 0.63247,
          "//": "Benchmark 以该项为准，越高越好↓：",
        "ndcg_at_10": 0.51669,
        "ndcg_at_100": 0.59061,
        "ndcg_at_1000": 0.61932,
        "ndcg_at_3": 0.55224,
        "ndcg_at_5": 0.52232,
        "precision_at_1": 0.63247,
        "precision_at_10": 0.25984,
        "precision_at_100": 0.03929,
        "precision_at_1000": 0.00466,
        "precision_at_3": 0.48479,
        "precision_at_5": 0.39134,
        "recall_at_1": 0.16121,
        "recall_at_10": 0.50435,
        "recall_at_100": 0.73425,
        "recall_at_1000": 0.87719,
        "recall_at_3": 0.32068,
        "recall_at_5": 0.4043
      },
      "mteb_dataset_name": "T2Retrieval",
      "mteb_version": "1.1.1"
    }
    ```

    -   分类（Classification）：（关注 `main_score` 项）

    ```json
    {
      "dataset_revision": null,
      "mteb_dataset_name": "Waimai",
      "mteb_version": "1.1.1",
      "test": {
        "accuracy": 0.8177,
        "accuracy_stderr": 0.020596358901514573,
        "ap": 0.6165018328568092,
        "ap_stderr": 0.02892173951174792,
        "evaluation_time": 2.47,
        "f1": 0.7980664755326708,
        "f1_stderr": 0.019041694919331758,
          "//": "Benchmark 以该项为准，越高越好↓：",
        "main_score": 0.8177
      }
    }
    ```

    -   重排序（Reranking）：（关注 `map` 项）

    ```json
    {
      "dataset_revision": null,
      "dev": {
        "evaluation_time": 712.57,
          "//": "Benchmark 以该项为准，越高越好↓：",
        "map": 0.6517172236785682,
        "mrr": 0.7483374708493192
      },
      "mteb_dataset_name": "T2Reranking",
      "mteb_version": "1.1.1"
    }
    ```

    -   STS：
        -   Pearson = 皮尔逊相关系数：衡量线性相关性
        -   Spearman = 斯皮尔曼相关系数：衡量排序相关性（A增加，B也增加/减小，则该相关系数越高）
        -   参考：[相关系数: Pearson vs Spearman - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/465213120)

    ```json
    {
      "dataset_revision": "6d1ba47164174a496b7fa5d3569dae26a6813b80",
      "mteb_dataset_name": "STS22",
      "mteb_version": "1.1.1",
      "test": {
        "evaluation_time": 1.45,
        "zh": {
          "//": "余弦相似度↓：",  
          "cos_sim": {
            "pearson": 0.46576773896833157,
            "//": "Benchmark 以该项为准，越高越好↓：",
            "spearman": 0.5535118570258535
          },
          "//": "欧几里得距离↓：",  
          "euclidean": {
            "pearson": 0.4274144004846264,
            "spearman": 0.5448514431273596
          },
          "//": "曼哈顿距离↓：",  
          "manhattan": {
            "pearson": 0.4688016767189237,
            "spearman": 0.5451646786859339
          }
        }
      }
    }
    ```

    -   聚类任务（Clustering）：

    ```json
    {
      "dataset_revision": null,
      "mteb_dataset_name": "CLSClusteringP2P",
      "mteb_version": "1.1.1",
      "test": {
        "evaluation_time": 709.48,
          "//": "Benchmark 以该项为准，越高越好↓：",
        "v_measure": 0.3430796134997738,
        "v_measure_std": 0.013224658125620643
      }
    }
    ```

    -   文本对分类（Pair Classification）：
        -   ap = 平均精确度 = Average Precision（越高越好）

    ```json
    {
      "dataset_revision": null,
      "mteb_dataset_name": "Ocnli",
      "mteb_version": "1.1.1",
      "validation": {
        "cos_sim": {
          "accuracy": 0.5798592311857066,
          "accuracy_threshold": 0.6471472978591919,
          "//": "Benchmark 以该类项为准↓：",
          "ap": 0.6040046859669777,
          "f1": 0.6830985915492958,
          "f1_threshold": 0.5322014689445496,
          "precision": 0.5425730267246737,
          "recall": 0.9218585005279831
        },
        "dot": {
          "accuracy": 0.5798592311857066,
          "accuracy_threshold": 0.6471472978591919,
          "ap": 0.6040025048845776,
          "f1": 0.6830985915492958,
          "f1_threshold": 0.5322014689445496,
          "precision": 0.5425730267246737,
          "recall": 0.9218585005279831
        },
        "euclidean": {
          "accuracy": 0.5798592311857066,
          "accuracy_threshold": 0.8400627374649048,
          "ap": 0.6040046859669777,
          "f1": 0.6830985915492958,
          "f1_threshold": 0.9672627449035645,
          "precision": 0.5425730267246737,
          "recall": 0.9218585005279831
        },
        "evaluation_time": 1.22,
        "manhattan": {
          "accuracy": 0.5820249052517596,
          "accuracy_threshold": 18.484230041503906,
          "ap": 0.6038369734249992,
          "f1": 0.6831683168316831,
          "f1_threshold": 21.856830596923828,
          "precision": 0.5342465753424658,
          "recall": 0.9472016895459345
        },
        "max": {
          "accuracy": 0.5820249052517596,
          "ap": 0.6040046859669777,
          "f1": 0.6831683168316831
        }
      }
    }
    ```

### 其他测评方法结果

[langchain(2)—基于开源embedding模型的中文向量效果测试 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/635670918)

[中文Sentence Embeddings text2vec-base-chinese VS OpenAIEmbedding - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/623912895)

## 智谱AI Embedding API 性能测试

加载文本：OReilly.Developing.Apps.with.GPT-4.and.ChatGPT.2023.9.pdf，共 261 页

根据 `batchSize = 512` 分割，共分割为 734 段文本，每页文档被分割为 2~5 段。

向量化总耗时：360181ms ≈ 6min

每段文本耗时区间：90%区间：350 ~ 450 ms；上限5%：耗时 600 ~ 900ms，上限1%：8 ~ 12s

重试次数：2/734

向量数据库存储耗时：883ms

