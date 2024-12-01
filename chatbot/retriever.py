from rank_bm25 import BM25Okapi
import numpy as np
import torch
import re
import string
from sentence_transformers import SentenceTransformer, util, CrossEncoder


DENSE_RETRIEVER_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
LLM_CORE_MODEL_NAME = "groq/llama3-8b-8192"


def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


class HybridRetrieverReranker:
    def __init__(self, dataset, dense_model_name=DENSE_RETRIEVER_MODEL_NAME, cross_encoder_model=CROSS_ENCODER_MODEL_NAME):
        if 'cleaned_text' not in dataset.columns:
            raise ValueError("Dataset must contain a 'cleaned_text' column.")

        self.dataset = dataset
        self.bm25_corpus = dataset['cleaned_text'].tolist()
        self.tokenized_corpus = [chunk.split() for chunk in self.bm25_corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        self.dense_model = SentenceTransformer(dense_model_name)
        self.cross_encoder = CrossEncoder(cross_encoder_model)


    def bm25_retrieve(self, query, top_k=70):
        """
        Retrieve top K documents using BM25.

        Args:
            query (str): Query text.
            top_k (int): Number of top BM25 documents to retrieve.

        Returns:
            list of dict: Top K BM25 results.
        """
        cleaned_query = clean_text(query)
        query_tokens = cleaned_query.split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        top_k_indices = np.argsort(bm25_scores)[::-1][:top_k]
        return self.dataset.iloc[top_k_indices].to_dict(orient='records')


    def dense_retrieve(self, query, candidates=None, top_n=35):
        """
        Retrieve top N documents using dense retrieval with LaBSE.

        Args:
            query (str): Query text.
            candidates (list of dict): Candidate documents from BM25.
            top_n (int): Number of top dense results to retrieve.

        Returns:
            list of dict: Top N dense results.
        """
        if candidates is None:
            candidates = self.dataset.to_dict(orient='records')
        query_embedding = self.dense_model.encode(query, convert_to_tensor=True)

        candidate_embeddings = torch.stack([
            eval(doc['chunk_embedding'].replace('tensor', 'torch.tensor')).clone().detach()
            for doc in candidates
        ])

        similarities = util.pytorch_cos_sim(query_embedding, candidate_embeddings).squeeze(0)
        top_n_indices = torch.topk(similarities, top_n).indices
        return [candidates[idx] for idx in top_n_indices]


    def rerank(self, query, candidates=None, top_n=3):
        """
        Rerank top documents using a CrossEncoder.

        Args:
            query (str): Query text.
            candidates (list of dict): Candidate documents from dense retriever.
            top_n (int): Number of top reranked results to return.

        Returns:
            list of dict: Top N reranked documents.
        """
        if candidates is None:
            candidates = self.dataset.to_dict(orient='records')
        query_document_pairs = [(query, doc['raw_text']) for doc in candidates]
        scores = self.cross_encoder.predict(query_document_pairs)
        top_n_indices = np.argsort(scores)[::-1][:top_n]
        return [candidates[idx] for idx in top_n_indices]



    def hybrid_retrieve(self, query, enable_bm25=True, enable_dense=True, enable_rerank=True, top_k_bm25=60, top_n_dense=30, top_n_rerank=2):
        """
        Perform hybrid retrieval: BM25 followed by dense retrieval and optional reranking.

        Args:
            query (str): Query text.
            top_k_bm25 (int): Number of top BM25 documents to retrieve.
            top_n_dense (int): Number of top dense results to retrieve.
            enable_dense (bool): Whether dense retrieval should be enabled.
            enable_rerank (bool): Whether reranking should be enabled.
            top_n_rerank (int): Number of top reranked documents to return.

        Returns:
            list of dict: Final top results after hybrid retrieval and reranking.
        """
        if enable_bm25:
            bm25_results = self.bm25_retrieve(query, top_k=top_k_bm25)
        else:
            bm25_results = None

        if enable_dense:
            dense_results = self.dense_retrieve(query, bm25_results, top_n=top_n_dense)
        else:
            dense_results = bm25_results

        if enable_rerank:
            final_results = self.rerank(query, dense_results, top_n=top_n_rerank)
        else:
            final_results = dense_results

        return final_results