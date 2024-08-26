import streamlit as st
import time
from API import *

# data = []
st.title("Thông tin chương trình Go Japan")
#data = None
# print(1)
with st.chat_message("assistant"):
    st.write("Tôi có thể giúp gì được cho bạn?")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:   
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something..."):
    with st.chat_message("user"):
        st.markdown(prompt)
        labels = retrieve_context_per_question(prompt, chunks_query_retriever)

    #Add message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        time.sleep(0.05)
        print(data)
        response, messages = run_openai_task(labels, prompt, data)
        # data.append({"role": "assistant", "content": response.content})
        data = messages
        data.append(response)
        # print(data)
        content = response.content
        st.markdown(content)

    st.session_state.messages.append({"role": "assistant", "content": content})
