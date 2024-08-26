import streamlit as st
import numpy as np
import time
import os

from openai import AzureOpenAI

AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT= os.getenv('AZURE_OPENAI_ENDPOINT')

data = []

# Get answer from prompt
def get_answer(prompt):
    client = AzureOpenAI(
      azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
      api_key=os.getenv("AZURE_OPENAI_API_KEY"),
      api_version="2024-02-01"
    )

    response = client.chat.completions.create(
        model="gpt-4o", # model = "deployment_name".
        messages=[
            {"role": "system", "content": prompt},
        ]
    )

    return response.choices[0].message.content

# Semantic chunking
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import AzureOpenAIEmbeddings
import nest_asyncio
from llama_parse import LlamaParse
from langchain_community.vectorstores import FAISS

import os
file_paths = ['data/Go_Japan.pdf']

for file_path in file_paths:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    
LLAMA_CLOUD_API_KEY = os.getenv('LLAMA_CLOUD_API_KEY')
nest_asyncio.apply()

documents = []
for file_path in file_paths:
    document = LlamaParse(
        result_type="text",
        language='vi',
        do_not_unroll_columns=True,
        max_timeout=2000).load_data(file_path)
    documents.append(document)

text_splitter = SemanticChunker(AzureOpenAIEmbeddings(), breakpoint_threshold_type='percentile', breakpoint_threshold_amount=90) # chose which embeddings and breakpoint type and threshold to use
for document in documents:
    docs = text_splitter.create_documents([document[i].text for i in range(len(document))])

embeddings = AzureOpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
chunks_query_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

def retrieve_context_per_question(question, chunks_query_retriever=chunks_query_retriever):
    """
    Retrieves relevant context and unique URLs for a given question using the chunks query retriever.

    Args:
        question: The question for which to retrieve context and URLs.

    Returns:
        A tuple containing:
        - A string with the concatenated content of relevant documents.
        - A list of unique URLs from the metadata of the relevant documents.
    """

    # Retrieve relevant documents for the given question
    docs = chunks_query_retriever.get_relevant_documents(question)

    # Concatenate document content
    # context = " ".join(doc.page_content for doc in docs)
    context = [doc.page_content for doc in docs]


    return context
def show_context(context):
    """
    Display the contents of the provided context list.

    Args:
        context (list): A list of context items to be displayed.

    Prints each context item in the list with a heading indicating its position.
    """
    for i, c in enumerate(context):
        print(f"Context {i+1}:")
        print(c)
        print("\n")

# Prompting

import json
from IPython.display import display, Markdown
from tenacity import retry, wait_random_exponential, stop_after_attempt
client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_API_ENDPOINT"),
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    api_version ="2024-02-01"
)

def system_message(labels):
    return f"""
    You are the organizer, so you know very well the information about this program.
    Your task is to provide information with evidence when receiving questions from users based on reference documents, and not say nonsense or you will be penalized.
    Your answer must be Vietnamese.
    In case user asks about the informations of questions which were answered before in history, try to use the messages before to answer it.  
    The answer is based on extracting information from: ({", ".join(labels)}).
    """

def assistant_message():
    return f"""
CASE DATA: 'HOẠT ĐỘNG NGOẠI KHÓA'
    EXAMPLE:
        Text: 'Tôi muốn đi chơi ở những nơi gần khu vực Nippori, Tokyo, với ngân sách dưới 5000 yên, hãy giới thiệu cho tôi những địa điểm tốt và phù hợp.'

        Step 1: Indentify key elements of the text:
            Địa điểm: Quanh khu vực Nippori, Tokyo
            Mục tiêu: Đi chơi
            Ngân sách: 5000 yên
            Bắt đầu: N/A

        Step 2: Finding places that are near to the desired location (Địa điểm)

        Step 3: Finding places that are suitable to the target (Mục tiêu)

        Step 4: Finding places that are suitable to the desired budget (Ngân sách)

        Step 5: Finding routes to go from the starting place, if none given, use Tokyo Station as the starting location

    OUTPUT STRUCTURE:
        Địa điểm: Công viên Ueno, Thư viện văn học thiếu nhi Quốc tế, 
        Ngân sách: Miễn phí - 600 yên
        Đường đi: Từ Tokyo Station, đi tàu JR Yamanote Line đến Nippori Station, đi bộ 5 phút


CASE DATA: 'GO JAPAN'

    EXAMPLE 1:
        Text: 'Tôi đang là một sinh viên năm 2 ngành Cơ Điện tử với trình độ tiếng Nhật N4, tôi có thể tham gia chương trình Go Japan không?'

        Step 1: Indentify main problem:
            'Tôi có thể tham gia chương trình Go Japan không?' -> 'Điều kiện tham gia chương trình Go Japan'
    
        Step 2: Indentify key elements of the text:
            Ngữ cảnh: Điều kiện tham gia chương trình Go Japan
            Đối tượng: Sinh viên năm 2 ngành Cơ Điện tử
            Trình độ tiếng Nhật: N4
    
        Step 3: Finding the relevant data for the problem, and indentify the answer:
            Đối tượng: Sinh viên tốt nghiệp 2024, 2025 -> Sinh viên năm 3, 4 -> 'Sinh viên năm 2' -> Không tham gia
            Trình độ tiếng Nhật: Từ N4 trở lên -> N4, N3, N2, N1 -> 'N4' -> Có thể tham gia
            Chuyên ngành:  Công nghệ thông tin, Khoa học máy tính, Kỹ thuật phần mềm, Điện tử viễn thông, Tự động hóa, Cơ điện tử -> 'Cơ Điện tử' -> Có thể tham gia
            (All conditions must be met in order to participate in the program)
        Step 4: Providing the answer to the problem, with proper reasoning:
            'Không thể tham gia chương trình vì bạn đang là năm 2, trong khi chương trình yêu cầu sinh viên năm 3 và 4 tham gia.'

    OUTPUT STRUCTURE:
        Không thể tham gia chương trình vì bạn đang là năm 2, trong khi chương trình yêu cầu sinh viên năm 3 và 4 tham gia,
        Relevant Data: 
            Đối tượng: Sinh viên tốt nghiệp 2024, 2025
            Trình độ tiếng Nhật: Từ N4 trở lên
            Chuyên ngành: Công nghệ thông tin, Khoa học máy tính, Kỹ thuật phần mềm, Điện tử viễn thông, Tự động hóa, Cơ điện tử

    EXAMPLE 2:
        Text: 'Giáo trình được sử dụng trong chương trình Go Japan là gì?'

        Step 1: Indentify main problem:
            'Giáo trình được sử dụng trong chương trình Go Japan là gì?' -> 'Giáo trình chương trình Go Japan'

        Step 2: Indentify key elements of the text:
            Ngữ cảnh: Giáo trình chương trình Go Japan

        Step 3: Finding the relevant data for the problem, and indentify the answer:
            Giáo trình: Chương trình học tập, học phần, sách giáo trình, tài liệu học tập -> 'Giáo trình' -> 'Sách giáo trình' -> 'Tài liệu học tập' -> 'Dekiru Nihongo N3/N2'

        Step 4: Providing the answer to the problem, with proper reasoning:
            'Giáo trình được sử dụng trong chương trình Go Japan là "Dekiru Nihongo N3/N2"'    

    OUTPUT STRUCTURE:
        Giáo trình được sử dụng trong chương trình Go Japan là 'Dekiru Nihongo N3/N2
        Relevant Data: 
            Giáo trình: Giáo trình tiếng Nhật

    EXAMPLE 3:
        Text: "Phương tiện ở Nhật đi ở làn bên trái hay bên phải?"

        Step 1: Indentify main problem:
            'Phương tiện ở Nhật đi ở làn bên trái hay bên phải?' -> 'Phương tiện ở Nhật'

        Step 2: Indentify key elements of the text:
            Ngữ cảnh: Phương tiện ở Nhật

        Step 3: Finding the relevant data for the problem, and indentify the answer:
            No relevant data found
        
        Step 4: Providing the answer to the problem, with proper reasoning:
            'Không tìm thấy dữ liệu liên quan'

    OUTPUT STRUCTURE:
        Không tìm thấy dữ liệu liên quan
        Relevant Data: 
            N/A

"""

def user_message(text):
    return f"""
TASK:
    Text: {text}
"""

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
def run_openai_task(labels, text, data):
    messages = data
    # um = None
    # sm = None
    messages.append({"role": "system", "content": system_message(labels=labels)})
    messages.append({"role": "assistant", "content": assistant_message()})
    messages.append({"role": "user", "content": user_message(text=text)})
    # print(messages)

    # TODO: functions and function_call are deprecated, need to be updated
    # See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools
    response = client.chat.completions.create(
        model="gpt-4o", # model = "deployment_name".
        messages=messages,
        top_p=0.9,
        #tools=generate_functions(labels),
        #tool_choice={"type": "function", "function" : {"name": "enrich_entities"}},
        temperature=0,
        frequency_penalty=0,
        presence_penalty=0
    )

    response_message = response.choices[0].message

    return response_message, messages
'''
test_query = """Chi tiết thời gian tham gia của chương trình"""
labels = retrieve_context_per_question(test_query, chunks_query_retriever)
show_context(labels)
result = run_openai_task(labels, test_query)
print(result)

#if "openai_model" not in st.session_state:
#    st.session_state["openai_model"] = "gpt-4o"
'''