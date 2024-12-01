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
    type="messages"
).launch()