from chatbot.retriever import HybridRetrieverReranker
from litellm import completion
import os
import ast

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DENSE_RETRIEVER_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"
LLM_CORE_MODEL_NAME = "groq/llama3-8b-8192"


class QuestionAnsweringBot:

    def __init__(self, docs, enable_bm25=True, enable_dense=True, enable_rerank=True, top_k_bm25=60, top_n_dense=30, top_n_rerank=2) -> None:
        self.retriever = HybridRetrieverReranker(docs)
        self.enable_bm25 = enable_bm25
        self.enable_dense = enable_dense
        self.enable_rerank = enable_rerank
        self.top_k_bm25=top_k_bm25
        self.top_n_dense=top_n_dense
        self.top_n_rerank=top_n_rerank

    def __get_answer__(self, question: str) -> str:
        PROMPT = """\
            You are an intelligent assistant designed to provide accurate and relevant answers based on the provided context.

            Rules:
            - Always analyze the provided context thoroughly before answering.
            - Respond with factual and concise information.
            - If context is ambiguous or insufficient or you can't find answer, say 'I don't know.'
            - Do not speculate or fabricate information beyond the provided context.
            - Follow user instructions on the response style(default style is detailed response if user didn't provide any specifications):
              - If the user asks for a detailed response, provide comprehensive explanations.
              - If the user requests brevity, give concise and to-the-point answers.
            - When applicable, summarize and synthesize information from the context to answer effectively.
            - Avoid using information outside the given context.
          """
        context = self.retriever.hybrid_retrieve(question,
                                                 enable_bm25=self.enable_bm25,
                                                 enable_dense=self.enable_dense,
                                                 enable_rerank=self.enable_rerank,
                                                 top_k_bm25=self.top_k_bm25,
                                                 top_n_dense=self.top_n_dense,
                                                 top_n_rerank=self.top_n_rerank
                                                 )

        context_text = [doc['raw_text'] for doc in context]

        response = completion(
                                model=LLM_CORE_MODEL_NAME,
                                temperature=0.0,
                                messages=[
                                    {"role": "system", "content": PROMPT},
                                    {"role": "user", "content": f"Context: {context_text}\nQuestion: {question}"}
                            ],
                            api_key=GROQ_API_KEY
                            )
        return response, context

    def form_response(self, question):
      llm_response, context = self.__get_answer__(question)

      metadata_raw = [doc['chapter_name'] for doc in context]
      metadata_cleaned = [ast.literal_eval(item) for item in metadata_raw]

      print('User:', question)
      print('System:', llm_response.choices[0].message.content)

      return f"**{llm_response.choices[0].message.content}**\n\nResources: {[chapter for doc in metadata_cleaned for chapter in doc]}"
