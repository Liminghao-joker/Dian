# 使用Word2Vec完成一个文本相似度搜索程序（gensim）
import gensim
from gensim.models import Word2Vec
from gensim import corpora
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np


class TextSimilarity:
    def __init__(self):
        self.text_corpus = [] # 存储原始语料库
        self.documents_vectors = [] # 存储文档向量
        self.model = None # Word2Vec模型

    def preprocess_text(self, text):
        """
        预处理文本，进行分词和小写化。
        """
        return gensim.utils.simple_preprocess(text)
    
    def train_model(self, corpus, vector_size=100, window=5, min_count=1, workers=4):
        """
        训练Word2Vec模型并计算文档向量。
        """
        self.text_corpus = corpus

        # 预处理
        processed_corpus = [self.preprocess_text(doc) for doc in corpus]

        # 训练模型
        self.model = Word2Vec(sentences=processed_corpus, vector_size=vector_size, window=window, min_count=min_count, workers=workers)

        # 计算每个文档的向量表示
        self._compute_document_vectors() # shape:(num_documents, vector_size)

    def _get_document_vector(self, text):
        """
        将文本转换成单个向量，返回平均向量。
        """
        words = self.preprocess_text(text)
        vectors = []
        for word in words:
            if word in self.model.wv: # 如果单词在词汇表中
                vectors.append(self.model.wv[word])
        
        if len(vectors) == 0:
            return None
        return np.mean(vectors, axis=0) # 对所有词向量取平均
    
    def _compute_document_vectors(self):
        """
        计算语料库中每个文档的向量表示。
        """
        self.documents_vectors = []
        for doc in self.text_corpus:
            vec = self._get_document_vector(doc)
            self.documents_vectors.append(vec)

    def search_similar(self, query, top_n=3):
        """
        搜索与查询文本最相似的top_n个文档。
        """
        query_vector = self._get_document_vector(query)
        if query_vector is None:
            return []

        similarities = []
        for idx, doc_vector in enumerate(self.documents_vectors):
            if doc_vector is None:
                continue
            sim = cosine_similarity([query_vector], [doc_vector])[0][0]
            similarities.append((self.text_corpus[idx], sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]

if __name__ == "__main__":
    corpus = [
        "Human machine interface for lab abc computer applications",
        "A survey of user opinion of computer system response time",
        "The EPS user interface management system",
        "System and human system engineering testing of EPS",
        "Relation of user perceived response time to error measurement",
        "The generation of random binary unordered trees",
        "The intersection graph of paths in trees",
        "Graph minors IV Widths of trees and well quasi ordering",
        "Graph minors A survey",
    ]

    query = "computer system response time"
    text_sim = TextSimilarity()
    text_sim.train_model(corpus)
    results = text_sim.search_similar(query, top_n=3)

    # 获取查询向量每个词的向量表示
    query_vector = text_sim._get_document_vector(query)
    print(f"\nAveraged Query Vector: {query_vector}\n")

    # if query_vector is not None:
    #     for word in text_sim.preprocess_text(query):
    #         if word in text_sim.model.wv:
    #             print(f"Word: {word} | Vector: {text_sim.model.wv[word]}")

    for doc, score in results:
        print(f"Document: {doc}\nSimilarity Score: {score}\n")