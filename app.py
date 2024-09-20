import streamlit as st
from agent import BiRAGAgent
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import utils
import time

st.set_page_config(layout="wide")

document_dict = {
    "오타니 쇼헤이": "docs_0.json"
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