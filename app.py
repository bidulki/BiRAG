import streamlit as st
from agent import BiRAGAgent
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import utils
import time

st.set_page_config(layout="wide")

document_dict = {
    "오타니 쇼헤이": "docs_0.json",
    "손흥민": "docs_1.json",
    "박지성": "docs_2.json",
    "크리스티아누 호날두": "docs_3.json",
    "켄 톰프슨": "docs_4.json",
    "데니스 리치": "docs_5.json",
    "크리스토퍼 콜럼버스": "docs_6.json",
    "니콜라 테슬라": "docs_7.json",
    "토마스 에디슨": "docs_8.json",
    "한강 (작가)": "docs_9.json",
    "삼성전자": "docs_10.json",
    "LG전자": "docs_11.json",
    "성균관대학교": "docs_12.json",
    "서울대학교": "docs_13.json",
    "다이제": "docs_14.json",
    "찰스 다윈": "docs_15.json",
    "마하트마 간디": "docs_16.json",
    "조지 워싱턴": "docs_17.json",
    "마틴 루터 킹": "docs_18.json",
    "SK 하이닉스": "docs_19.json"
}

def load_explorer(faiss_index_path, embedding_model):
    explorer = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)
    return explorer

@st.cache_resource
def get_agent():
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    explorer = load_explorer("./DB/faiss_index", embedding_model)
    return BiRAGAgent(explorer, document_dict)

font_size = 30

st.markdown(
    f'<span style="font-size:{font_size}px;">Read and Write RAG: rwRAG</span>',
    unsafe_allow_html=True,
)
st.markdown("----")

agent = get_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message['content'])

if prompt := st.chat_input(placeholder="입력"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    utils.format_and_print_user_input(prompt)
    response = agent(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        if type(response) == str:
            utils.print_log("Received string response")
            assistant_response = response
            full_response += assistant_response + " "
            message_placeholder.markdown(full_response, unsafe_allow_html=True)
            utils.format_and_print_genai_response(full_response)
        else:
            utils.print_log("Received stream response")
            for chunk in response:
                if isinstance(chunk, str):
                    full_response += chunk
                    time.sleep(0.05)
                elif chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                
                message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
            
            utils.format_and_print_genai_response(full_response)
            print(full_response)
            assistant_message = agent.make_message("assistant", full_response)
            agent.history.append(assistant_message)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})