import numpy as np
from typing import List

from FlagEmbedding import BGEM3FlagModel
from pymilvus import (
    MilvusClient,
    DataType,
    AnnSearchRequest,
    RRFRanker,
)
from langchain_core.documents import Document

from config import EMBEDDING_MODEL_PATH, MILVUS_DB_FILE, MILVUS_COLLECTION, DENSE_DIM, BGE_BATCH_SIZE
from src.client.mongodb_config import MongoConfig


class BGERetriever:

    def __init__(self, docs: List[Document], force_rebuild: bool = False):
        # load model
        self._model = BGEM3FlagModel(EMBEDDING_MODEL_PATH, use_fp16=True)

        # connect to milvus-lite
        self._client = MilvusClient(MILVUS_DB_FILE)
        self._mongo  = MongoConfig.get_collection("manual_text")

        collection_exists = self._client.has_collection(MILVUS_COLLECTION)
        if collection_exists and not force_rebuild:
            print(f"Collection '{MILVUS_COLLECTION}' already exists, skipping build.")
        else:
            self._build(docs)

    # --- public ---

    def retrieve_topk(self, query: str, topk: int = 10) -> List[Document]:
        """Hybrid search (dense + sparse) with RRF, fetch full docs from MongoDB."""
        q = self._model.encode_queries(
            [query],
            return_dense=True,
            return_sparse=True,
        )
        dense_query  = q["dense_vecs"][0].tolist()
        sparse_query = self._sparse_to_milvus(q["lexical_weights"][0])

        dense_req = AnnSearchRequest(
            data         = [dense_query],
            anns_field   = "dense_vector",
            param        = {"metric_type": "COSINE"},
            limit        = topk,
        )
        sparse_req = AnnSearchRequest(
            data         = [sparse_query],
            anns_field   = "sparse_vector",
            param        = {"metric_type": "IP"},
            limit        = topk,
        )

        results = self._client.hybrid_search(
            collection_name = MILVUS_COLLECTION,
            reqs            = [dense_req, sparse_req],
            ranker          = RRFRanker(k=60),
            limit           = topk,
            output_fields   = ["unique_id"],
        )[0]

        docs = []
        for hit in results:
            unique_id = hit["entity"]["unique_id"]
            record    = self._mongo.find_one({"unique_id": unique_id})
            if record:
                docs.append(Document(
                    page_content = record["page_content"],
                    metadata     = record["metadata"],
                ))
        return docs

    # --- private ---

    def _build(self, docs: List[Document]) -> None:
        """Drop existing collection, create schema, encode and insert all docs."""
        if self._client.has_collection(MILVUS_COLLECTION):
            self._client.drop_collection(MILVUS_COLLECTION)

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("unique_id",     DataType.VARCHAR,            max_length=64, is_primary=True)
        schema.add_field("dense_vector",  DataType.FLOAT_VECTOR,       dim=DENSE_DIM)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

        index_params = self._client.prepare_index_params()
        index_params.add_index("dense_vector",  index_type="AUTOINDEX",             metric_type="COSINE")
        index_params.add_index("sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

        self._client.create_collection(
            collection_name = MILVUS_COLLECTION,
            schema          = schema,
            index_params    = index_params,
        )
        print(f"Collection '{MILVUS_COLLECTION}' created.")

        # encode + insert in batches
        texts      = [doc.page_content      for doc in docs]
        unique_ids = [doc.metadata["unique_id"] for doc in docs]

        total = 0
        for i in range(0, len(docs), BGE_BATCH_SIZE):
            batch_texts = texts[i : i + BGE_BATCH_SIZE]
            batch_ids   = unique_ids[i : i + BGE_BATCH_SIZE]

            out = self._encode(batch_texts)

            entities = [
                {
                    "unique_id":     uid,
                    "dense_vector":  dvec.tolist(),
                    "sparse_vector": self._sparse_to_milvus(svec),
                }
                for uid, dvec, svec in zip(batch_ids, out["dense"], out["sparse"])
            ]
            self._client.insert(MILVUS_COLLECTION, entities)
            total += len(entities)
            print(f"  Inserted {total}/{len(docs)} chunks...")

        print(f"Build complete: {total} chunks indexed.")

    def _encode(self, texts: List[str]) -> dict:
        """Encode a batch of texts, return dense array and sparse list."""
        out = self._model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            batch_size=BGE_BATCH_SIZE,
        )
        return {
            "dense":  out["dense_vecs"],        # np.array (N, 1024)
            "sparse": out["lexical_weights"],   # list of dicts {token_id: weight}
        }

    @staticmethod
    def _sparse_to_milvus(lexical_weights: dict) -> dict:
        """Convert bge-m3 lexical_weights {token_id: weight} to Milvus sparse format."""
        return {int(k): float(v) for k, v in lexical_weights.items()}


if __name__ == "__main__":
    from src.client.mongodb_config import MongoConfig

    col  = MongoConfig.get_collection("manual_text")
    docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in col.find()]

    retriever = BGERetriever(docs)
    results   = retriever.retrieve_topk("How to Adjust the Shoulder Anchor Height", topk=3)
    for r in results:
        print(f"Page: {r.metadata.get('page')}")
        print(r.page_content)
        print("=" * 60)