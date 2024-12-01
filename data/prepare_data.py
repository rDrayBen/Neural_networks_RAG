import os
import re
import string
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import CharacterTextSplitter
import pandas as pd

DATA_FILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "CountMonteCristoFull.txt")
DENSE_RETRIEVER_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
LLM_CORE_MODEL_NAME = "groq/llama3-8b-8192"

with open(DATA_FILE_PATH, "r", encoding="utf-8") as f:
    data_corpus = f.read()

splitter = CharacterTextSplitter(separator="\n\n", chunk_size=10_000, chunk_overlap=1_000)
text_chunks = splitter.create_documents([data_corpus])

prev_chapter_name = ''
for chunk in text_chunks:
    chunk.metadata['belongs_to'] = set()
    curr_chapter_name = ''
    index_start_chapter_name = chunk.page_content.find('Chapter')

    if index_start_chapter_name == -1:
        curr_chapter_name = prev_chapter_name
    else:
        # if prev_chapter_name is not empty and next chapter start further than first 40% of the chunk.
        # This means that the name of the prev chapter isn't in this chunk, but relevant info can be found.
        if prev_chapter_name != '' and index_start_chapter_name > int(len(chunk.page_content) * 0.4):
            chunk.metadata['belongs_to'].add(prev_chapter_name)

        index_end_chapter_name = chunk.page_content.find('\n\n', index_start_chapter_name)
        curr_chapter_name = chunk.page_content[index_start_chapter_name:index_end_chapter_name]
        prev_chapter_name = curr_chapter_name
    chunk.metadata['belongs_to'].add(curr_chapter_name)

    chunk.metadata['belongs_to'] = list(chunk.metadata['belongs_to'])


def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

dense_model = SentenceTransformer(DENSE_RETRIEVER_MODEL_NAME)

def calculate_embeddings(text):
    return dense_model.encode(text, convert_to_tensor=True)

chunked_data_corpus = []

for index, chunk in enumerate(text_chunks):
    chunked_data_corpus.append({
        'raw_text': chunk.page_content,
        'cleaned_text': clean_text(chunk.page_content),
        'chunk_embedding': calculate_embeddings(chunk.page_content),
        'chapter_name': chunk.metadata['belongs_to']
    })

chunked_data_corpus_df = pd.DataFrame(chunked_data_corpus)

chunked_data_corpus_df.to_csv('chunked_data_corpus.csv', index=False)