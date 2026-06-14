1.实现BPE
    Naive BPE
    Fast BPE
2. 实现WordPiece
     Naive Wordiece
     Fast 

| 挑战 | 具体内容 |
|---|---|
| **效率优化** | 从 naive → fast 实现，并能用清晰的方式解释加速原理 |
| **歧义消解** | 同一个词可能有多种切分方式，掌握两种算法各自的消歧策略 |
| **比较与评估** | 设计有信息量的方式，比较两种算法的词表和分词结果，并与形态学分词进行对比 |

需要读的文献:
Sennrich et al. 2016 —— BPE 在 NLP 中的奠基论文
Schuster & Nakajima 2012 —— WordPiece 的原始来源（日语/韩语语音识别）
HuggingFace 文档 —— 两种算法的现代讲解，适合对照实现
Mielke et al. 2021 —— 开放词表建模和分词的历史综述，提供更宏观视角

简单来说：
手写 BPE 和 WordPiece，先写能跑的版本，再优化到能处理真实数据的高效版本，最后设计实验比较两者——包括与形态学分词的对比。

语料库：
测试阶段：
- WikiText-103 HuggingFace datasets获取
搭配 CELEX 或 MorphoLex 作为英语形态学gold standard，用于评估分词是否切在真实形态边界上