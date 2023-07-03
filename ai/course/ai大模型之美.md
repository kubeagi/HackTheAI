- https://time.geekbang.com/column/intro/100541001?tab=catalog
- 推荐参考链接：
	- 李沐精讲论文系列 https://www.bilibili.com/video/BV1AF411b7xQ/
	- chatgpt官方文档
		- chatgpt能做什么，openai的示例 https://platform.openai.com/examples/
		- chatgpt 模型列表 https://platform.openai.com/docs/models/overview
		- chatgpt API 接口 https://platform.openai.com/docs/api-reference
	- 一些实践
		- 如何根据医学论文来回答问题 https://mp.weixin.qq.com/s/iplUoK_JYeL_9EC7Ttt3tw
		- LangChain AI Handbook https://www.pinecone.io/learn/langchain/
- 之前面对的问题是“有多少人工就有多少智能”，所有的 “智能” 都来自于大量的人工数据标注，甚至很多硬编码的业务规则 => 人工智能是人工智障
- 为什么人人都应该学习如何开发新一代 AI 应用？
	- 第一个原因，是这一轮的 AI 浪潮里，开发新的 AI 应用的门槛大大降低了。
		- 任何一个稍有开发经验的工程师，都能够在几周甚至几天之内，学会使用这些基础模型以及相应的开放 API 开发出有使用价值的应用。
	- 第二个原因，是这一轮的 AI 浪潮里，对应技术能够应用的范围非常广泛，可以说是包罗万象。
		- 之前，对于每一个具体问题我们都要单独收集数据、训练单独的机器学习模型来解决某一个小问题。
		- 这一轮的 AI 浪潮开始让我们看见了 “通用人工智能”（AGI）的雏形，AI 应用的覆盖领域被大大扩展了，几乎任何一个问题都有机会通过 AI 来解决优化。今天你所在的行业和领域，都有机会通过简单的 AI 应用开发，提升效率和产出。
	- 第三个原因，是这个浪潮带来的变化会对我们每一个人的工作带来巨大的冲击。
- 具体使用
	- 情感分析（一段文字是正面还是负面）
		- 之前的套路：按分类问题处理，人工标注，训练模型，测试。
		- 例子 https://www.kaggle.com/code/ankumagawa/sentimental-analysis-using-naive-bayes-classifier
		- 传统方法挑战：
			- 很多话，词组一样，句子不同，情感也不同，解决办法是特征工程：去掉停用词，去低频词等等
			- 机器学习的经验：需要将数据集切分成训练（Training）、验证（Validation）、测试（Test）三组数据，然后通过 AUC 或者混淆矩阵（Confusion Matrix）来衡量效果
		- 大模型：
			- 使用 Embedding API：计算文本的向量值，和 “好评” 和 “差评” 这两个字向量比较即可。
				- 和开源对比：Facebook的Fasttext，Google的T5，结论是T5比Fasttext好一些。T5 Base（2.2亿参数）好于T5 Small（6千万参数）但是还是不如Embedding API。
				- 不同embedding 模型的维度不一样，最少1024，最多1024*12，名称都是计算机史上名人。类似docker的默认容器名
				- 坑：
					- openai接口限制数据长度。
						- 需要开发自己用openai的tiktoken库算一下token数，注意选择的编码方式和使用模型一致 https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
					- openai 对API限速。
						- backoff库；
						- 使用openai的batch接口调用
						- 但是openai也会限制每分钟可以处理的token数，所以批量调用也有限制。
					- 建议不要存csv，存parquet序列化格式。
				- 通过 OpenAI 的 API 获取到 Embedding，然后通过一些简单的线性模型，我们就能获得很好的分类效果。
					- 机器学习基础概念
						- 准确率，代表模型判定属于这个分类的标题里面判断正确的有多少
						- 召回率，代表模型判定属于这个分类的标题占实际这个分类下所有标题的比例，也就是没有漏掉的比例
						- 综合考虑这两项得出的结果，就是 F1  分数（F1 Score）。F1 分数，是准确率和召回率的调和平均数
						- 支持的样本量，是指数据里面，实际是这个分类的数据条数有多少。
			- 零样本分类，也就是我们不需要任何新的样本来训练机器学习的模型，就能进行分类。我们认为，之前经过预训练的大语言模型里面，已经蕴含了情感分析的知识。我们只需要简单利用大语言模型里面知道的 “好评” 和 “差评” 的概念信息，就能判断出它从未见过的评论到底是好评还是差评。
	- 写文案
		- 一般用 Completion 补全接口
			- temperature参数定义了相同prompt返回结果的随机性，越大越随机。越小每次返回的结果越固定，一般越严肃的场合应该设定越小。
			- max_tokens prompt和AI返回的接口最大多少token，和费用有关。
			- n 能返回多少条回答，默认1
			- stop，输出遇到什么值停止，和费用有关。默认空，最多放4个选项。比如`\n\n`说明要新建一段了。
			- presence_penalty，如果一个token在前面内容出现过，后面生成时给予一定概率的惩罚，让AI聊新内容。
			- frequency_penalty，对重复出现的Token进行概率惩罚，让AI尽量用不同的表述。
			- logit_bias 不让出现的词，给个-100。一定出现的词，给个100
	- 对话
		- 用 completions 接口
			- 具体例子看官方的吧 https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
			- 使用方便，但费用可能比想象的高，发送和接收的内容都收费哦
		- 界面一般用 gradio 库
			- 基本是python代码，还能在 jupyter Notebook 中显示，对开发友好
			- 被HuggingFace收购，可以直接免费部署。
		- 多轮对话
			- 我们只要把之前对话的内容也都放到prompt里面，把整个上下文都提供给 AI。AI 就能够自动根据上下文，回答第二个问题。（很多跳过gpt的“越狱”，诱发gpt说违规内容就是改这里的上文实现的）
			- 上面的[情感分析](todo)也可以这么处理，`判断一下用户的评论情感上是正面的还是负面的\n评论：xxxx \n 情感：正面\n 评论：yyy\n 情感：负面\n评论：输入内容\n情感：`
			- "给一个任务描述、给少数几个例子、给需要解决的问题” 这样三个步骤的组合，Few-Shots Learning（少样本学习），也就是给一个或者少数几个例子，AI 就能够举一反三，回答我们的问题。
			- 为了避免超token和费用太高，方法有：
				- 只传输最近几次的对话。太早的对话丢弃。或者限制token数，截取最后的多少个。
	- 文本聚类 （把文本根据相似度分类）
		- 用 Embdding 把文本转为向量，然后向量聚类，比如 K-Means 算法
		- 实践注意：
			- 中间结果保存
			- 可以让openai对聚合的类命名
			- 同样含义下，中文消耗token比英文多。生产建议英文prompt，最多让 gpt `generate Chines`
		- 可以用于[多轮对话]，让AI对过去几轮对话做个100字的总结，作为新的prompt的一部分，开始下一轮对话。
		- OpenAI提供 Moderate接口 https://platform.openai.com/docs/api-reference/moderations ，对输入内容检查
	- 优化搜索 / 推荐
		- 传统搜索依赖 Elasticsearch 先分词，再倒排索引，缺陷是同义词不好找。可能需要研发手工加同义词表
		- 现在可以把用户的搜索词Embedding转为向量，和数据中现有的向量计算距离。这个套路也能用于推荐冷启动。
		- 技巧：
			- 甚至可以让openai生成测试数据，比如fake的淘宝标题
			- 检索向量，可以用向量数据库pinecone或weaviate 或 milvus 或者Faiss包，内存中相似性搜索。
	- 读书/读文：将外部资料变成索引
		- ![](https://static001.geekbang.org/resource/image/e4/6c/e4ae3fbeaa82d82e317dfcef40679f6c.jpg?wh=1896x1292)
		- bing的可能解法：先搜索，后prompt，把搜索词最接近的几条也发给gpt，让AI回答。
		- python库 llama-index，也是向openai发请求，但是封装了很多方法。
			```bash
			  Context information is below. 
			  ---------------------
			  {context_str} #根据向量距离自动算出来的和问题相关的上下文内容
			  ---------------------
			  Given the context information and not prior knowledge, answer the question: {query_str}
			```
			- 对书总结：
				- 按内容拆分，每段内容小结，组成树形结构再小结即可。llama-index内置这个方法。
			- 对图片总结
				- 支持多模态，把图片转为文字再处理。官方例子是拍吃饭小票，问哪天吃了什么吃了多少钱 https://github.com/jerryjliu/llama_index/blob/main/examples/multimodal/Multimodal.ipynb
			- 其他，https://llama-hub-ui.vercel.app/ 包括pdf，mongo，youtube等等。
	- 开源节流
		- 用 google 的 Colab  菜单`runtime`->`Change runtime type`改成 gpu。gpu 一定额度内免费。
		- 使用 HuggingfaceEmbedding 库
			- 使用 HuggingFace 的好处是，你可以通过一套代码使用所有的 transfomers 类型的模型。
		- 单机模型的一个限制是token不能太大。上下文不够。
		- 清华的 chatglm 模型，Chinese-llama-alpaca模型，链家的belle模型
	- 写EXCEL 插件，主要是思路，最终是一段VBA代码。
	- 写单元测试。
		- 用多步prompt，例子 https://github.com/openai/openai-cookbook/blob/main/examples/Unit_test_writing_using_a_multi-step_prompt.ipynb
		- 先让AI解释这个函数，然后让AI写计划，打算加哪些单元测试，最后执行，输出这些单元测试。
			- 解释
				```bash
				  How to write great unit tests with {unit_test_package}
				  In this advanced tutorial for experts, we'll use Python 3.10 and `{unit_test_package}` to write a suite of unit tests to verify the behavior of the following function.
				  """
				  {function_to_test}
				  """
				  Before writing any unit tests, let's review what each element of the function is doing exactly and what the author's intentions may have been.
				  First,
				```
			- 计划
				```bash
				  先把第一步的prompt和回答放到最前面，再加下面的内容，一并发给openai
				  A good unit test suite should aim to:
				  - Test the function's behavior for a wide range of possible inputs
				  - Test edge cases that the author may not have foreseen
				  - Take advantage of the features of `{unit_test_package}` to make the tests easy to write and maintain
				  - Be easy to read and understand, with clean code and descriptive names
				  - Be deterministic, so that the tests always pass or fail in the same way
				  
				  `{unit_test_package}` has many convenient features that make it easy to write and maintain unit tests. We'll use them to write unit tests for the function above.
				  
				  For this particular function, we'll want our unit tests to handle the following diverse scenarios (and under each scenario, we include a few examples as sub-bullets):
				```
				- 这一步如果openai生成的测试少，就在让他输出一些
					```bash
					  In addition to the scenarios above, we'll also want to make sure we don't forget to test rare or unexpected edge cases (and under each edge case, we include a few examples as sub-bullets):-
					```
			- 生成
				```bash
				  前面全部，再加下面的内容，一并发送给openai
				  Before going into the individual tests, let's first look at the complete suite of unit tests as a cohesive whole. We've added helpful comments to explain what each line does.
				  """python
				  import {unit_test_package}  # used for our unit tests
				  
				  {function_to_test}
				  """
				  Below, each test case is represented by a tuple passed to the @pytest.mark.parametrize decorator
				```
			- 最后用ast检查是否有语法错误。出错从头再来一次。
	- langchain
		- 比如上面写单元测试需要的模式，做个抽象，就会有langchain这个库的使用。
		- 链式调用
			- 比如问问题：把问题翻译为英语->英语提问->英语回答->翻译为中文。
		- 还能调用外部系统，类似 chatgpt 的plugins
			- 比如算数学问题直接问chatgpt出来的结果是错的，可以让ai输出python函数，然后plugin为执行python函数，得到最后结果。 https://python.langchain.com/docs/modules/chains/additional/llm_math
			- 比如使用 LLMRequestsChain，通过一个 HTTP 请求来得到问题的答案。最简单粗暴的一个办法，就是直接通过一个 HTTP 请求来问一下 Google ![](https://static001.geekbang.org/resource/image/e2/45/e29294012c309f3f808693cea5216445.png?wh=1920x2181)
			- 通过 VectorDBQA 来实现先搜索再回复的能力
		- 记忆功能
			- 整个对话上下文叫 memory。只保留几轮，总结前面的内容，或者两者的结合，都已经固化为langchain的函数。除此之外，还有一定要记住的EntityMemory函数，把memory放到redis，放到外部存储的方法等等。
		- 支持多种功能，中介和特工 Agent
			- 先写多个单功能的工具，然后让AI自动选择这个问题应该用哪个工具完成，比如电商可能有导购资讯，售中咨询查订单快递，FAQ常见问题答疑这3个工具。
			- 这种分而治之，就是agent类
		- 模型微调 Fine tune
			- openai的一个接口，输入内容类似
				```bash
				  
				  {"prompt": "<prompt text>", "completion": "<ideal generated text>"}
				  {"prompt": "<prompt text>", "completion": "<ideal generated text>"}
				  {"prompt": "<prompt text>", "completion": "<ideal generated text>"}
				  ...
				```
			- 会返回一个新模型，调用这个模型费用大概是原模型的5到10倍，同时，训练模型也会收费。
			- 还能在这个微调上继续微调。
- 语音和视频的应用

