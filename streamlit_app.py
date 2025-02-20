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
text = "В Северной столице несколько сотрудников одного из медучреждений покинули свои рабочие места. В четверг, 30 апреля, из Покровской больницы Петербурга уволились четверо врачей второго кардиологического отделения. В комитете по здравоохранению «Форпосту» сообщили, что больница уже нашла им замену. Лечение больных продолжается Точная причина ухода медиков не называется. Однако в СМИ появилась информация, что трое из уволившихся участвовали в видеообращении, в котором сотрудники лечебницы сообщали об отсутствии средств защиты в их стационаре. После этого, руководство медучрежения якобы пригрозило работникам последствиями, что стало последней каплей на фоне сложной ситуации в больнице. Напомним, в начале апреля группа врачей петербургской Покровки  записала  ролик с жалобой на нехватку защитной экипировки и кислородных баллонов. Тогда в городском комитете по здравоохранению опровергли данную информацию, сообщив, что для лечения пациентов с пневмонией не требуется особого оснащения докторов. Спустя две недели медработники  пожаловались  на неготовность учреждения к приёму заражённых COVID-19. Они попросили власти города присвоить Покровской больнице статус инфекционной, поскольку они фактически работают с больными коронавирусом."
result = st.button("Сгенерировать заголовок")

if result:
    input = tokenize_text(tokenizer, text)
    output = model.generate(model_input)
    output_str = tokenizer.decode(model_output.squeeze(),
                                  skip_special_tokens=True)
    st.write("Заголовок:")
    st.write(output_str)
