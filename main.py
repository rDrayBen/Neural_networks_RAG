import gradio as gr
import pandas as pd
from chatbot.bot import QuestionAnsweringBot
import os

chunked_data_corpus_df = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data\\chunked_data_corpus.csv"))

bot = QuestionAnsweringBot(chunked_data_corpus_df)

def message_respond(message, history):
    answer = bot.form_response(message)
    return answer

gr.ChatInterface(
    fn=message_respond,
    type="messages",
    title="RAG System for 'The Count of Monte Cristo' book",
    description="Here you can ask any questions in the context of the book 'The Count of Monte Cristo'.",
    theme=gr.themes.Monochrome(font='Lora', text_size='lg', radius_size='sm'),
    examples=["Who is Monte Cristo?", "What is the title of Chapter 93?", "Why Edmond Dantes was in prison?", "How many years does Edmon Dantes spent in prison?"],
    cache_examples=False,
).launch()