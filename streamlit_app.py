import streamlit as st
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration


@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer():
    model = T5ForConditionalGeneration.from_pretrained("IlyaGusev/rut5_base_headline_gen_telegram")
    tokenizer = AutoTokenizer.from_pretrained("IlyaGusev/rut5_base_headline_gen_telegram")

    return model, tokenizer

def tokenize_text(tokenizer, text):
    input = tokenizer(text,
                      truncation=True,
                      max_length=512,
                      return_tensors="pt")

    return input["input_ids"].unsqueeze(0)

model, tokenizer = load_model_and_tokenizer()

st.title("Генерация заголовков для новостных текстов")
text = st.text_area(label="Введите текст новости:")
st.write(text)
result = st.button("Сгенерировать заголовок")

if result:
    input = tokenize_text(text)
    output = model.generate(model_input)
    output_str = tokenizer.decode(model_output.squeeze(),
                                  skip_special_tokens=True)
    st.write("Заголовок:")
    st.write(output_str)
