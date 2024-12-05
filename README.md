# Neural_Networks_RAG

This project is aimed at developing RAG system for documents corpus. To be more precise, this system operates with the english version of the book 'The Count of Monte Cristo'.

* Folder **experiments** contains .ipynb file with all the details about each part of the RAG system as well as comments about finding the best solutions for each task.

* Folder **data** contains such files:
  1. CountMonteCristoFull.txt - original text of the book.
  2. chunked_data_corpus.csv - dataset with columns = ['raw_text'(raw text divided into chunks), 'cleaned_text'(raw text chunks cleaned from extra spaces and regexes), 'embedding'(embeddings for raw_text), 'chapter_name'(names of chapters that this particular chunk of text belongs to)].
  3. prepare_data.py - file that performs dataset preparations.

* Folder **chatbot** contains such files:
  1. bot.py - file with LLM calling class
  2. retiriever.py - implementation of hybrid retriever(bm25 + bi-encoder)

* main.py is the file that implements UI and launches the system.

To start the system you need to install required libraries via:
```
pip install -r requirements.txt
```

Then, RAG system can be launched via:
```
python3 main.py
```

The flow of the system is provided further:
1. Acquire chunked dataset.
2. Find 20% of the most relevant documents from the whole corpus using bm25.
3. Take 50% of the most relevant documents retrieved on the step 2 using bi-encoder('all-MiniLM-L6-v2').
4. Out of all the documents retrieved on the step 3 take 2 most relevant, using cross-encoder('cross-encoder/ms-marco-MiniLM-L-12-v2').
5. Use these 2 documents for models context.

Deployed RAG system can be found on HuggingFace spaces via this link: https://huggingface.co/spaces/RabotiahovDmytro/RAGSystem.
