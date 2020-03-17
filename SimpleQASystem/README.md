# 搭建一个简单的问答系统(英文)

## 说明
#### 首先这是一个基于问答库的问答系统，并不是你想象的siri和小娜之类，当然，那些也是由语料库训练的。
#### 根据问答库里的问答，对于相近的问题，系统会给出相应的答案，例如:
#### 原问题(存在问答库的问题) How many volts does a track lighting system usually use?
#### 对应答案：12 or 24 volts
#### 对于上面的问题，让我们换一种说法:
#### How many volts we need if we want to build a track lighting system?
#### 会同样给出答案：12 or 24 volts
#### 所以只要你的问答库足够大，这个系统就足够智能。

## 原理
#### 整个系统的处理流程如下：
#### 1. 文本预处理技术（词过滤，标准化）
#####   		a. 停用词过滤   
#####   		b. 转换成小写   
#####   		c. 去掉一些无用的符号
#####   		d. 去掉出现频率很低的词
#####   		e. 对于数字的处理： 123 -> "#number"
#####   		f. stemming 词干提取
#### 2. 文本的表示（tf-idf)
#### 3. 文本相似度计算
#####   		a. 采用余弦相似度
#### 4. 文本高效检索
#####   		a. 倒排表 https://en.wikipedia.org/wiki/Inverted_index 


