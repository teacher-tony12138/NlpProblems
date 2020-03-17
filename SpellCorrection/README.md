# 句子拼写纠错(英文)

## 示例:
#### French Finance Mrnistei Edouard Baelladur, meanwhile,  confirmed there would be a communique at the end of the  mnetieg.
#### Mrnistei -> Minister
#### Baelladur -> Balladur
#### mnetieg -> meeting

## 原理
#### 基于noisy channel model https://en.wikipedia.org/wiki/Noisy_channel_model
#### 假设正确单词为c，错误单词为e，那么我们的问题就变成：在给定错误单词的情况下，
#### 计算使得 P(c|e) 最大的单词，而 P(c|e) = P(e|c)*P(c)/P(e), 由于给定的拼写错误单词，P(e) = 1
#### P(c|e) = P(e|c)*P(c)
#### P(e|c)的意思是给定正确单词c，拼写错误为e的概率，我们可以从比如搜索日志中获取，例如我们经常在搜索时遇到：
#### 你是否想搜xxx，这里xxx就是系统提示的正确单词
#### P(c)是正确单词的概率，比如apple可能是appl的一种正确拼写，但是app也可能是一种正确的拼写。
#### 我们可以借助语言模型来描述这个概率,语言模型我们可以借助nltk或其他语料库来构建.(https://en.wikipedia.org/wiki/Language_model) 



