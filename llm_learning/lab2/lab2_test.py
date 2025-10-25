# 带打印说明的 Word2Vec 文本相似度演示脚本
# 该脚本基于 lab2_1025.py 的 TextSimilarity 类，并在每个关键函数中打印出结果/中间信息，
# 便于理解程序的功能结构和数据流。

from lab2 import TextSimilarity
import numpy as np


class VerboseTextSimilarity(TextSimilarity):
    def preprocess_text(self, text):
        tokens = super().preprocess_text(text)
        print(f"[preprocess_text] input='{text}' -> tokens={tokens}")
        return tokens

    def train_model(self, corpus, vector_size=100, window=5, min_count=1, workers=4):
        print("[train_model] start training model...")
        super().train_model(corpus, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
        vocab_size = len(self.model.wv.key_to_index)
        sample_words = list(self.model.wv.key_to_index.keys())[:10]
        print(
            f"[train_model] done. vector_size={self.model.wv.vector_size}, "
            f"window={window}, min_count={min_count}, vocab_size={vocab_size}"
        )
        print(f"[train_model] sample vocab (first 10): {sample_words}")
        print(f"[train_model] computed document vectors: {len(self.documents_vectors)} items")

    def _get_document_vector(self, text):
        # 这里故意调用父类逻辑前后打印，便于观察覆盖的词及输出向量形状
        tokens = super().preprocess_text(text)
        covered = []
        vectors = []
        if self.model is None:
            print("[_get_document_vector] model is None, please train first.")
            return None
        for w in tokens:
            if w in self.model.wv:
                covered.append(w)
                vectors.append(self.model.wv[w])
        if not vectors:
            print(f"[_get_document_vector] input='{text}' -> no covered tokens in vocab, return None")
            return None
        vec = np.mean(vectors, axis=0)
        print(
            f"[_get_document_vector] input='{text}' -> covered_tokens={covered} -> "
            f"vec_shape={vec.shape}, vec_preview={np.round(vec[:5], 4)}"
        )
        return vec

    def _compute_document_vectors(self):
        print("[_compute_document_vectors] start computing document vectors...")
        self.documents_vectors = []
        for i, doc in enumerate(self.text_corpus):
            vec = self._get_document_vector(doc)
            self.documents_vectors.append(vec)
            shape = None if vec is None else vec.shape
            print(f"  doc[{i}] vector shape={shape}")
        print("[_compute_document_vectors] done.")

    def search_similar(self, query, top_n=3):
        print(f"[search_similar] query='{query}', top_n={top_n}")
        result = super().search_similar(query, top_n)
        print("[search_similar] results:")
        for doc, score in result:
            print(f"  score={score:.4f} | {doc}")
        return result


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

    print("=== Step 1: init ===")
    ts = VerboseTextSimilarity()

    print("\n=== Step 2: preprocess query (standalone) ===")
    ts.preprocess_text(query)

    print("\n=== Step 3: train model on corpus ===")
    ts.train_model(corpus)

    print("\n=== Step 4: compute query vector ===")
    qv = ts._get_document_vector(query)
    print(f"[main] query vector shape: {None if qv is None else qv.shape}")

    print("\n=== Step 5: search similar documents ===")
    ts.search_similar(query, top_n=3)
