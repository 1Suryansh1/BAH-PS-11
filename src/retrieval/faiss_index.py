import faiss
import numpy as np
import time

class FAISSRetrievalIndex:
    def __init__(self, embedding_dim: int = 256, 
                 index_type: str = 'IVFFlat',
                 n_list: int = 100):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.n_list = n_list
        self.index = None

    def build(self, embeddings: np.ndarray, labels: np.ndarray, sample_ids: list = None):
        D = self.embedding_dim
        if self.index_type == 'IVFFlat':
            quantizer = faiss.IndexFlatIP(D)
            self.index = faiss.IndexIVFFlat(quantizer, D, self.n_list, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)
            self.index.add(embeddings)
        elif self.index_type == 'IVFPQ':
            quantizer = faiss.IndexFlatIP(D)
            self.index = faiss.IndexIVFPQ(quantizer, D, self.n_list, 8, 8)
            self.index.train(embeddings)
            self.index.add(embeddings)
        else:
            self.index = faiss.IndexFlatIP(D)
            self.index.add(embeddings)

        self.labels = labels
        self.sample_ids = sample_ids

    def search(self, query: np.ndarray, k: int = 10):
        if self.index is None:
            raise ValueError("Index not built.")
        distances, indices = self.index.search(query, k)
        ret_labels = [self.labels[idx] for idx in indices]
        return distances, indices, ret_labels

    def benchmark_speed(self, n_queries: int = 1000):
        if self.index is None:
            raise RuntimeError("Index not built.")
        queries = np.random.randn(n_queries, self.embedding_dim).astype(np.float32)
        faiss.normalize_L2(queries)
        
        start = time.time()
        self.index.search(queries, 10)
        end = time.time()
        
        total_time = (end - start) * 1000 # ms
        mean_time = total_time / n_queries
        throughput = n_queries / (end - start)
        
        print(f"Query time: {mean_time:.3f} ms/query, Throughput: {throughput:.2f} queries/sec")
