# Lab1 IMDB数据集

## 学习目标

1. 了解IMDB数据集
2. 学习文本预处理的基础操作，包括HTML标签清理、大小写转换、标点处理等
3. 掌握词表(Vocabulary)的构建原理和方法，理解词表与词典的区别
4. 学会使用Dataset，Dataloader
5. 使用Dataset和Dataloader包装IMDB数据集

## 实验内容

1. **数据读取与处理**
	1. 控制数据集规模，随机选取2000条评论
2. **词表构建** 
	1. 分词统计
	2. 过滤低频率词汇 `min_count`
	3. 排序，分配唯一索引
	4. 特殊标记`<unk>`
3. **序列转换** **`text_to_sequence`**

## 学习产出

`lab11.py`

`lab12.py`

`label11_result.txt`

`label12_result.txt`

## 参考资料

1. [认识IMDB数据集](https://blog.csdn.net/Delusional/article/details/113357449?ops_request_misc=%7B%22request%5Fid%22%3A%223ec79fd1fdbf4817bad2e72779f8ed6c%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=3ec79fd1fdbf4817bad2e72779f8ed6c&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-113357449-null-null.142^v102^pc_search_result_base6&utm_term=imdb数据集&spm=1018.2226.3001.4187)
2. [IMDB数据集下载](http://ai.stanford.edu/~amaas/data/sentiment/)
3. [什么是词表、词典？](https://www.rethink.fun/chapter12/词典生成.html)
4. [Dataset, Dataloader详解](https://blog.csdn.net/junsuyiji/article/details/127585300?utm_medium=distribute.pc_feed_404.none-task-blog-2~default~BlogCommendFromBaidu~Rate-1-127585300-blog-null.262^v1^pc_404_mixedpudn&depth_1-utm_source=distribute.pc_feed_404.none-task-blog-2~default~BlogCommendFromBaidu~Rate-1-127585300-blog-null.262^v1^pc_404_mixedpud)