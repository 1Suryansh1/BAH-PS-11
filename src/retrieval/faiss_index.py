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
        self.labels = None
        self.sample_ids = None

    def build(self, embeddings: np.ndarray, labels: np.ndarray, sample_ids: list = None):
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        
        D = self.embedding_dim
        N = embeddings.shape[0]

        if self.index_type == 'IVFPQ':
            assert D % 8 == 0, f"embedding_dim {D} must be divisible by 8 for IVFPQ"

        # Centroids constraint fallback for small datasets
        index_type = self.index_type
        if index_type in ['IVFFlat', 'IVFPQ'] and N < self.n_list:
            print(f"Warning: Database size {N} is smaller than IVF centroids {self.n_list}. Falling back to IndexFlatIP.")
            index_type = 'Flat'
            
        if index_type == 'IVFFlat':
            quantizer = faiss.IndexFlatIP(D)
            self.index = faiss.IndexIVFFlat(quantizer, D, self.n_list, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)
            self.index.add(embeddings)
        elif index_type == 'IVFPQ':
            quantizer = faiss.IndexFlatIP(D)
            self.index = faiss.IndexIVFPQ(quantizer, D, self.n_list, 8, 8)
            self.index.train(embeddings)
            self.index.add(embeddings)
        else:
            self.index = faiss.IndexFlatIP(D)
            self.index.add(embeddings)

        self.labels = labels
        self.sample_ids = sample_ids

    def search(self, query: np.ndarray, k: int = 10, query_indices: np.ndarray = None):
        """
        Args:
            query: (N_query, D) query embeddings
            k: number of nearest neighbors to retrieve
            query_indices: (N_query,) optional indices of queries inside database to filter self-matches.
        """
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")
            
        query = np.ascontiguousarray(query, dtype=np.float32)
        faiss.normalize_L2(query)

        fetch_k = k + 1 if query_indices is not None else k
        distances, indices = self.index.search(query, fetch_k)
        
        filtered_distances = []
        filtered_indices = []
        ret_labels = []
        
        for i in range(len(query)):
            d_i = distances[i]
            idx_i = indices[i]
            
            if query_indices is not None:
                self_idx = query_indices[i]
                mask = (idx_i != self_idx)
                d_i = d_i[mask][:k]
                idx_i = idx_i[mask][:k]
            else:
                d_i = d_i[:k]
                idx_i = idx_i[:k]
                
            filtered_distances.append(d_i)
            filtered_indices.append(idx_i)
            
            labels_i = []
            for idx in idx_i:
                if idx == -1:
                    if isinstance(self.labels, np.ndarray):
                        labels_i.append(np.zeros_like(self.labels[0]))
                    else:
                        labels_i.append(None)
                else:
                    labels_i.append(self.labels[idx])
            ret_labels.append(labels_i)
            
        return np.array(filtered_distances), np.array(filtered_indices), ret_labels

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
