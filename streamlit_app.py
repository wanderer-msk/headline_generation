import streamlit as st
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

def load_model_and_tokenizer():
    model = T5ForConditionalGeneration.from_pretrained("wanderer-msk/ruT5-base_headline_generation")
    tokenizer = AutoTokenizer.from_pretrained("wanderer-msk/ruT5-base_headline_generation")

    return model, tokenizer

def tokenize_text(tokenizer, text):
    input = tokenizer(text,
                      truncation=True,
                      max_length=512,
                      return_tensors="pt")

    return input["input_ids"]

model, tokenizer = load_model_and_tokenizer()

st.title("Генерация заголовков для новостных текстов")
text = st.text_area(label="Введите текст новости:")
result = st.button("Сгенерировать заголовок")

if result:
    input = tokenize_text(tokenizer, text)
    output = model.generate(input)
    output_str = tokenizer.decode(output.squeeze(),
                                  skip_special_tokens=True)
    st.write("Заголовок:")
    st.write(output_str)
