# RWKV 模型

源代码: https://github.com/BlinkDL/RWKV-LM

论文预印本: https://arxiv.org/abs/2305.13048

## 介绍

>   RWKV 是基于循环神经网络的大语言模型算法，其性能与类 transformer 的大语言模型（如 GPT）相媲美，亦可实现并行训练，结合了两类模型的特点：表现优异、快速训练推理、节约显存、“无限”上下文长度、自由调整语句 embedding（特征向量）。
>

分类：模型 - 语言模型 - 算法

中文介绍：https://zhuanlan.zhihu.com/p/603840957

## 特点

- 模型、算法、数据集完全开源，且开源模型里性能表现最好：https://zhuanlan.zhihu.com/p/629786540
- “无限”上下文 —— 意味着对话长度不受限制（GPT 类 chatbot 单轮对话长度有一定限制，超过则可能会忘记之前的内容）
- 推理所需算力小，7B 参数模型仅需约 3GB 显存即可在本地运行，亦可使用 CPU 进行推理
- 基于本地推理的特性，可通过插件实现类似 Stable Diffusion 的精调（fine-tuning），满足个性化自定义需求（参见下列应用）
- 	作者的目标：“做语言模型的 Stable Diffusion”
- 	**可直接商用**

## 实践

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

## 性能测试记录

### Case 1 Intel 11代移动端 CPU 推理

-   系统配置：Intel i7-1135G7 4c8t，内存 DDR4 16G@3200MHz
-   资源占用：6G 内存，推理期间 CPU 占用率 30%
-   模型：RWKV-4 全球语言 1.5B 参数量，2.94GB
-   推理速度：约 4 字/秒
-   能力表现：可完成简单的问答、翻译功能；虚构场景型任务表现良好（如撰写机器人导游的介绍词）
-   问题：真实世界知识类问答不严谨，如书籍作者名错误，人物与事迹移花接木等；长回复中出现语言混杂，如中文回应一段后忽而变成英文（Claude 有类似现象）；前后文关联问题，如提出 AI 话题后若提出其他问题，回答倾向于与 AI 相关话题直接靠拢，而非区别开来。上述问题可能是由于参数量较小及采样参数设置问题导致。

### Case 2 Nvidia T4 GPU 推理

-   系统配置：Nvidia T4 16G
-   资源占用：约 8G 显存，推理期间显卡接近满载
-   模型：RWKV-4 全球语言 3B 参数量，~7GB
-   推理速度：约 20 字/秒
-   能力表现：可以完成约 5~10 轮左右的对话问答功能，并可以实现简单的角色扮演对话；补全/指令类任务表现尚可；
-   问题：补全类任务中有一定概率出现话题跳脱的情况，例如要求生成对话，但忽然开始产出一段类似知识百科条目的内容，该问题可能需要通过限制单次 token 产生量规避；对话类任务中有一定概率出现自言自语，或不断重复对话的情况，概率随对话轮次的提高而提高，且与对话内容主题可能有一定关联性；单次生成内容过长，则前后文有一定概率失去关联性，例如过长的代码。

### Case 3 Nvidia RTX 4090 GPU 推理

-   系统配置：Nvidia RTX 4090 24G
-   资源占用：约 16G 显存，推理期间显卡占用 80%
-   模型：RWKV-4 全球语言 7B 参数量，~14B，开启 CUDA 编译
-   推理速度：约 30~40 字/秒
-   能力表现：可以完成较长的对话功能；角色扮演的性格内容更加丰富；补全一般知识类说明文效果良好，准确度较高。
-   问题：重复对话及长补全内容跳脱的问题仍有概率出现，需要调整相关推理参数以达到最好效果。

## 3B on T4 表现测试记录

### 任务 1 英文知识类生成

指令：`$ +gen Here is a brief introduction of the company Microsoft`

3B on T4：

>   Microsoft Corporation is an American multinational technology company, based in Redmond, Washington. It was founded by Bill Gates and Paul Allen on April 4, 1975. Microsoft develops and sells computer software, consumer electronics and personal computers. It also provides server services through its subsidiary company Microsoft Azure.
>   With more than 500 million active users in over 180 countries worldwide, it is the world’s largest software company by revenue and number of users.
>
>   1 What Is The Business Model Of Microsoft?
>   2 About Microsoft
>   3 The Brands Of Microsoft
>   4 How Does Microsoft Make Money?
>   5 How Much Is The Net Worth Of Microsoft?
>   6 Where Can I Find More Information About This Company?
>
>   What Is The Business Model Of Microsoft?
>   The business model of the company is based on two major products: Office and Windows operating system. The company has been a leader in this field for decades and has developed many successful products that are used by millions of people around the world. In addition to these products, the company also offers other services such as cloud computing, data storage, video conferencing services and more. As a result of its innovative approach to business models, it has become one of the most successful companies in the world today. If you’re interested in *(Bob: +++)* learning more about this company and its business model, then keep reading!
>
>   The Brands Of Microsoft
>   Microsoft is a global technology company that offers a wide range of products and services. The company’s brand names include Windows, Office, Xbox, Skype, Bing and more. Microsoft has been around for over 50 years and continues to grow and innovate with each passing day. Here are some of the most popular products from Microsoft:
>   Office is a suite of productivity software that includes word processing, spreadsheets, presentations and databases. It was first released in 1989 and has since become one of the most widely used office suites in the world.
>   Windows is a family of graphical operating systems developed by Microsoft for personal computers (PCs). Windows XP was released on July 22nd 2001 and replaced Windows ME as Microsoft’s flagship operating system before being replaced by Windows 7 on October 22nd 2009.
>   Skype is an international communication platform that allows users to make voice calls over the internet using their computer or mobile device. It was first released on March 25th 2003 and has since become one of the most popular applications on smartphones worldwide.
>   Bing is an internet search engine that provides users with access to millions of web pages from all over the world. It was founded by Paul Allen in 1998 and acquired by Microsoft in 2009 for $26.2 billion.
>   Xbox is a video game console developed by Microsoft. It was first released on November 15th 2001 and has since become one of the most popular gaming consoles on the market today.
>   Microsoft Office is a suite of productivity software developed by Microsoft. It includes Word, Excel, PowerPoint, Outlook and Access among other applications. Office is available on Windows operating systems (Windows XP through Windows 10), macOS (since version 10), iOS (since version 9) and Android (since version 4).
>
>   How Does Microsoft Make Money?
>   Microsoft is a global technology company that offers a wide range of products and services to consumers, businesses, governments and organizations around the world. The company’s primary product lines include personal computers, servers, mobile devices, software development tools and cloud computing services.
>   The company’s revenue comes from various sources including licensing fees for its software products, hardware sales from its hardware division (such as Xbox), subscriptions for its cloud computing services (such as Office 365) and advertising revenues generated through targeted marketing campaigns directed at specific audiences or segments of society such as teenagers or gamers/sports fans/students/business professionals etc.. Additionally Microsoft also generates revenue through its investments in other companies such as Microsoft Research Labs which conducts research into areas such as artificial intelligence and machine learning among others.
>
>   How Much Is The Net Worth Of Microsoft?
>   Microsoft is one of the most valuable technology companies in the world, with a market capitalization of over $1 trillion. The company’s net worth is based on its stock price, which fluctuates daily due to fluctuations in the market value of its shares. However, it’s safe to say that Microsoft has been one of the most successful companies in recent years.
>
>   Where Can I Find More Information About This Company?
>   Microsoft is one of the world’s leading software companies, with a wide range of products and services that are used by businesses and individuals alike. If you’re looking for more information about this company, there are several ways to find out what it does and how it makes money. First, check out their website at www.microsoft.com – this should give you a good overview of what they do and how they make money from their products and services. Alternatively, you can also visit their investor relations page at www.msftinvestor.com – here you’ll find all kinds of financial data about the company itself as well as information about its stock prices over time! Finally, don’t forget to check out their YouTube channel – there are plenty of videos featuring interviews with key executives at Microsoft that provide valuable insights into how they operate day-to-day operations as well as their vision for the future!

评价：自动生成了包含数个章节的介绍长文，涵盖了相当一部分角度，文章前后整体结构连贯，内容准确度尚可；结语出现了风格类似油管博主的自荐台词；给出的大纲与实际文案对应不上，漏了一节；若落地作为实际使用文案，需要人工作进一步精修、事实核验，但总的来说效率与质量相当不错。

### 任务 2 编程能力

指令：`+gen Here is the python code of an AVL tree:`

3B on T4：

```
    class AVLTree:
        def __init__(self):
            self.root = None

        def insert(self, key, value):
            """Inserts a new node into the tree."""
            if not self.root:
                self.root = AVLNode(key, value)
            else:
                mid = (self.root.key + self.root.value) // 2
                if key < mid:
                    self.insert(key, value)
                elif key > mid:
                    self.insert(mid+1, value)
        
        def find_kth_largest(self, k):
            """Finds the kth largest element in the tree."""
            return self._find_kth_largest(0, k-1)
        
        def _find_kth_largest(self, i, k):
            if not self.root:
                return None
        
            # Find the kth smallest element in the subtree rooted at this node.
            k -= 1  # Assumes a binary search tree is implemented using an array of size 1<<k
        
            if k == 0:
                return None
        
            # Traverse down the tree until we find a node smaller than the root node.
            while k >= 0 and self.root
        .key < self.root.value:
                    if k % 2 == 0:
                        k -= 1  # Go down one level in the tree
                    else:
                        k -= 1  # Go up one level in the tree

                i = i // 2
        
            return self._find_kth_largest(i, k)
        
        def insert(self, key, value):
            """Inserts a new node into the tree."""
            if not self.root:
                self.root = AVLNode(key, value)
            else:
                mid = (self.root.key + self.root.value) // 2
        
                if key < mid:
                    self._insert_in_right(key, value, mid)
                elif key > mid:
                    self._insert_in_left(key, value, mid+1)
        
        def _find_kth_largest(self, i):
            """Finds the kth largest element in the tree."""
            if not self.root:
                return None
        
            # Find the kth smallest element in the subtree rooted at this node.
            k = -1
            k += 1  # Assumes a binary search tree is implemented using an array of size 1<<k
        
            while k >= (,,,)
```

评价：测试不同语言，仍出现类似以上的自我重复现象，效果较差。

指令（伪代码）：`+gen Here is the pseudo code of an AVL tree:`

3B on T4：

```
AVL_TREE_T* createAVLTree(int val) {
    AVL_TREE_T *root = new AVL_TREE_T;
    root->left = NULL;
    root->right = NULL;
    root->val = val;
    return root;
}

void insertNode(AVL_TREE_T *tree, int key, int value) {
    if (tree == NULL)
        return;

    if (key < tree->val) {
        if (tree->left == NULL) {
            tree->left = new AVL_TREE_T();
            tree->left->val = key;
            tree->left->right = NULL;
            tree->right = NULL;
        } else {
            insertNode(tree->left, key, value);
        }
    } else if (key > tree->val) {
        if (tree->right == NULL) {
            tree->right = new AVL_TREE_T();
            tree->right->val = key;
            tree->right->left = NULL;
            tree->left = NULL;
        } else {
            insertNode(tree->right, key, value);
        }
    }
}
```

评价：尝试令其生成伪代码以减少单次输出量，得到的代码结果质量有一定提升，但无法达到输出即可用。

总结：3B World 模型难以胜任算法或代码类任务：生成途中常常跳出类似 Stackoverflow 的无关问答内容；即便拆解问题依然很难实现开箱即用；可能需要考虑 Raven 类英文专精模型的效果（例如 Raven 14B）

### 任务 3 中文知识类生成

指令1：`+gen 以下是书籍《计算机程序的构造和解释》的简要介绍：`

3B on T4：

>   《计算机程序的构造和解释》是一本介绍计算机程序设计的经典书籍。该书由著名的计算机科学家约翰·麦卡锡（John McCarthy）所著，共分为六个部分，涵盖了计算机程序设计的各个方面。
>   第一部分介绍了计算机编程语言，包括变量、数据类型、控制流程等概念。该部分对于初学者来说比较容易理解。
>   第二部分主要介绍了如何使用编译器将源代码转换为可执行代码。这一部分对于想要开发自己的软件或者系统的人非常有帮助。
>   第三部分讲述了操作系统、数据库和网络编程等内容。这些内容是现代计算机科学中最重要的部分之一，因为它们构成了我们日常生活中不可或缺的基础。
>   第四部分介绍了算法、数据结构和操作系统等内容。这些内容对于想要成为一名优秀的程序员非常重要。
>   第五部分主要介绍了编译器、调试器和性能分析等内容。这些内容对于想要深入了解计算机编程的人非常有帮助。
>   第六部分是一个完整的工程项目，包括使用计算机语言设计一个基本的游戏。这个项目将需要学习一些计算机科学的基础知识，如数据结构、算法和逻辑思维等。
>   总之，《计算机程序的构造和解释》是一本很好的教材，它可以帮助初学者理解计算机编程中最基本的概念和原则。

评价：相当的驴唇不对马嘴（汗），作者名错误（实际为人工智能领域学者），内容更是完全不沾边而更像是 CSAPP 的内容。但若不经过事实核验则看上去很像一回事。

指令2：`+gen 以下是腾讯公司的介绍：`

3B on T4：

>   腾讯公司是一家全球领先的互联网科技公司，总部位于中国深圳。它成立于1998年，目前拥有多个业务板块，包括社交、游戏、电子商务和云计算等。腾讯在全球范围内拥有超过1.5亿用户，其中包括Facebook、微信和Instagram等知名社交媒体平台。
>   腾讯的主要业务板块如下：
>   1. 社交：腾讯旗下的微信、QQ和QQ空间是中国最大的社交网络平台之一，拥有超过20亿注册用户。
>   2. 游戏：腾讯旗下的暴雪娱乐、暴雪奇幻之旅和守望先锋等游戏品牌在全球范围内都拥有庞大的用户群体。
>   3. 电子商务：腾讯旗下的淘宝、天猫和京东等电子商务平台已经成为全球最大的电子商务平台之一。
>   4. 云计算：腾讯旗下的阿里云、华为云等企业级服务平台在全球范围内拥有广泛的用户基础。
>
>   总之，腾讯是一家综合性的科技公司，其主要业务板块涵盖了社交、游戏、电子商务和云计算等领域。

评价：只有大方向尚且符合事实，细节部分同样出现了张冠李戴、货不对板的现象。

总结：3B 模型在中文知识类生成任务的表现非常一般，特别是细节部分容易出现串词、张冠李戴等事实错误。