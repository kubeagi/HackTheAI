## 介绍
可以为大量非结构化数据的存储、查询和分析提供更具可扩展性的效率优势的解决方案。  
**Embedding Database 需要使用模型将数据转换为向量。**

### Embedding 技术

`针对问题`: 文本、图像、音频等非结构数据存储问题。  
`解决方法`: 利用Embedding技术将高维度的数据（例如文字、图片、
音频）映射到低维度空间，即把图片、声音和文字转化为向量来表
示，将这些向量存储起来就构成向量数据库。实现Embedding过程
的⽅法包括神经⽹络、LSH（局部敏感哈希算法）等。

### 向量索引技术

`针对问题`: 向量数据维度很高，直接进行全量扫描或者基于树结构
的索引会导致效率低下或者内存爆炸。  
`解决方法`: 采用近似搜索算法来加速向量的检索，通常利用向量之
间的距离或者相似度来检索出与查询向量相近的K个向量，距离度量
包括欧式距离、余弦、内积、海明距离，向量索引技术包括 k-d
tree（k-dimensional tree）, PQ（乘积量化）, HNSW（可导航小
世界网络）等。

---

## Chroma

默认使用 `all-MiniLM-L6-v2`模型进行数据到向量的转换。支持的[模型列表](https://www.sbert.net/docs/pretrained_models.html)。目前该数据库只有`python`和`js` 的 sdk。

有三种部署方式：

- 直接使用内存
- 将数据持久化到文件
- 生产环境下，部署一个后端服务，然后 chroma 的 client 去连。

该数据库可以直接与 `langchain` 结合，有相关的开发方法指导。但是需要注意，chroma 有一个数据收集的能力，可以自动关闭。收集的信息主要包括：

- chroma 的版本和环境信息。
- 收集 add 和 delete 命令的信息。

代码是 python 的，没看到其他的协议相关介绍，所以如果配合 `langchain`, 应该使用 python 进行开发。

## Faiss

Faiss 还能够运行在 CPU 和 GPU 两种环境中，同时可以使用 C++ 或者 Python 进行调用。Faiss 可以通过选择不同的索引方式和模型平衡搜索速度和精确度。

Faiss 支持的[索引模式列表](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)

这是对Faiss基本使用的一篇介绍[文章](https://soulteary.com/2022/09/03/vector-database-guide-talk-about-the-similarity-retrieval-technology-from-metaverse-big-company-faiss.html)，理解Faiss
