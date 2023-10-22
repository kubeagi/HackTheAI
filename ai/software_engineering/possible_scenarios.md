如何使用 AIGC 提升产品/项目的研发效率和质量，更好的实现降本增效，结合提示词工程（Prompt Engineering），这里列举一下可能的应用场景：

* 辅助写邮件
收集一些非敏感/脱敏后的邮件或者模版，作为私域数据向量化，结合大模型进行服务

* 文章总结，概括大意

* 文章翻译（文档、英文阅读、写作参考等）

* 需求系统模拟及反馈，需求文档生成

* 伪代码转为可运行代码

* 代码/文档质量
1. 自动生成注释
2. 单元测试
3. Personally Identifiable Information 敏感信息检测，日志/代码等

* 内部知识库建设

* 文档质量评估/项目反馈评估
比如基于以下 Prompt 的评级：

1）阅读以下文章摘录，并根据以下标准提供反馈：语法、清晰度、连贯性、论证质量和证据使用。为每个属性提供1-10的评分，并附上评分的理由。
"基于微前端框架、低代码开发，定义了标准的组件封装及发布模式，让开发者可以在底座之上按照开发规范进行组件的快速开发及部署，并在统一的服务门户上对外提供服务"

* API 生成/模拟

* 结合低代码框架，辅助前端快速开发

其他可以通过 GPT 实现软件工程任务自动化的提示词模式：
| 任务分类  | 任务描述 |
|---------|---------|
| Requirements Elicitation   | Requirements Simulator<br>Specification Disambiguation<br>Change Request Simulation  |
| System Design and Simulation   | API Generator <br>API Simulator<br> Few-shot Example Generator <br>Domain-Specific Language (DSL) <br>Creation Architectural Possibilities  |
| Code Quality   | Code Clustering<br>IntermediateAbstraction<br>Principled Code<br>Hidden Assumptions  |
| Refactoring   | Pseudo-code Refactoring<br>Data-guided Refactoring  |

