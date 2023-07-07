## Chroma 使用

根据官方文档给的例子，进行一个简单的操作

```shell
(chroma-langchain-tutorial) (base) ➜  chroma-langchain-tutorial git:(main) ✗ python3
Python 3.10.10 (main, Mar 21 2023, 18:45:11) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import chromadb
>>> chroma_client = chromadb.Client()
Using embedded DuckDB without persistence: data will be transient
>>> collection = chroma_client.create_collection(name="my_collection")
No embedding_function provided, using default embedding function: SentenceTransformerEmbeddingFunction
>>> collection.add(
...     documents=["This is a document", "This is another document"],
...     metadatas=[{"source": "my_source"}, {"source": "my_source"}],
...     ids=["id1", "id2"]
... )
>>> results = collection.query(
...     query_texts=["This is a query document"],
...     n_results=2
... )
>>> results
{'ids': [['id1', 'id2']], 'embeddings': None, 'documents': [['This is a document', 'This is another document']], 'metadatas': [[{'source': 'my_source'}, {'source': 'my_source'}]], 'distances': [[0.7111214399337769, 1.0109777450561523]]}
>>> 

```

简单将手动写的两份文档写入到向量数据。

## langchain

对于大批量文档，pdf, 图片等， `langchain/document_loaders` 提供了多种Load数据的方式，选择一个合适的即可。

Loader 会得到一个 `List[Document]` 的数据, 这个 `Document` 就是 `{string, metadata}` ，这个就是包含完整的文档内容，以及一份元数据。

然后通过 `langchain.text_splitter RecursiveCharacterTextSplitter` 通过 `\n\n`, `\n`, `""`, `" "` 将数据分割成字符串列表。例如
`{string:"abce defg", metadata:= {"id":"o"}}`

分割后得到的两份文档`[{string:"abce", metadata:{"id":"o"}}, {string:"defg", metadata: {"id":"o"}}]`

得到上述文档后， 根据 https://github.com/hwchase17/langchain/tree/master/langchain/embeddings 根据这里提供的多种数据入库模型，选择一个，就可以将数据放进去


## langchain-chatglm


api.py 是整个web服务的入口，定义了若干api
其中有update_file的接口，这个就是会根据上传的文件类型，
这个接口会调用 `init_knowledge_vector_store` 里面会有 `load_file` 选择不同的loader，得到数据后，在做 `text_split` 将数据入库。

