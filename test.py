import faiss
import numpy as np


d = 1024
ncentroids = 50
vectors1 = np.random.random((1000, d))
vectors2 = np.random.random((5, d))
query = vectors2
k = 3
ids = np.random.randint(0, 10, size=(5))
print('ids', ids)

quantizer = faiss.IndexFlatL2(d)


# faiss.IndexIVFFlat demonstration
index_ivf_flat = faiss.IndexIVFFlat(quantizer, d, ncentroids)
index_ivf_flat.train(vectors1)
index_ivf_flat.add_with_ids(vectors2, ids)
index_ivf_flat.nprobe = 5
D_ivf_flat, I_ivf_flat = index_ivf_flat.search(query, k)

print('distances ivf flat', D_ivf_flat)
print('indices ivf flat', I_ivf_flat)


# faiss.IndexPQ demonstration
nbits = 2
num_subspaces = 4

assert d % num_subspaces == 0

index_pq = faiss.IndexPQ(d, num_subspaces, nbits)
index_pq_ids = faiss.IndexIDMap(index_pq)
index_pq_ids.train(vectors1)
index_pq_ids.add_with_ids(vectors2, ids)
D_pq, I_pq = index_pq_ids.search(query, k)

print('distances pq', D_pq)
print('indices pq', I_pq)


# faiss.IndexIVFPQ demonstration

# Note the original C++ function declaration with parameters:
# IndexIVFPQ::IndexIVFPQ(
#             Index* quantizer,
#             size_t d,
#             size_t nlist,
#             size_t M,
#             size_t nbits_per_idx,
#             MetricType metric)
#             : IndexIVF(quantizer, d, nlist, 0, metric), pq(d, M, nbits_per_idx)

#                                for IVF       for IVF    for IVF + PQ   for PQ
index_ivf_pq = faiss.IndexIVFPQ(quantizer, d, ncentroids, num_subspaces, nbits)
index_ivf_pq.train(vectors1)
index_ivf_pq.add_with_ids(vectors2, ids)
index_ivf_pq.nprobe = 5
D_ivf_pq, I_ivf_pq = index_ivf_pq.search(query, k)

print('distnaces ivf pq', D_ivf_pq)
print('indices ivf pq', I_ivf_pq)

print('type of ivf_pq', type(index_ivf_pq))
