from bertopic import BERTopic
from hdbscan import HDBSCAN
from langchain.chains import ConversationalRetrievalChain

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma

import streamlit as st
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms.ollama import Ollama
from umap import UMAP

from preprocessing import preprocess_documents
from read_docs import read_pdf_files, read_context

st.set_page_config(page_title="Discovery RAG")
st.title("Discovery RAG")


llm  = Ollama(model="mistral", verbose=True, temperature=0)


def topic_modeling():
    umap_model = UMAP(n_neighbors=2, n_components=5, min_dist=0.0, metric='cosine')
    hdbscan_model = HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_method='eom',prediction_data=True)
    st.session_state.topic_model = BERTopic(hdbscan_model=hdbscan_model, umap_model=umap_model)
    pdf_list = read_pdf_files(st.session_state.folder_path)
    docs = preprocess_documents(pdf_list)
    topics, probabilities = st.session_state.topic_model.fit_transform(docs)



def rag():
    context = " ".join(st.session_state.topic_model.representative_docs_[int(st.session_state.topic_n)])

    docs = read_context(context, chunk_size=500, chunk_overlap=50)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(documents=docs, embedding=embedding_model)

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    st.session_state.c_retrieval = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(),
                                                        memory=memory)


with st.sidebar:

    st.session_state.folder_path = st.text_input(value="./docs", label="select path of your docs here")
    x=st.button("Submit Documents", on_click=topic_modeling)
    st.session_state.topic_n = st.text_input(value="0", label="with which set of documents you want to chat?",
                                             on_change=rag)
if x:
    st.plotly_chart(st.session_state.topic_model.visualize_topics())
if "messages" not in st.session_state:
    st.session_state.messages = []
#
for message in st.session_state.messages:
    st.chat_message('human').write(message[0])
    st.chat_message('ai').write(message[1])
    #
if query := st.chat_input():
    st.chat_message("human").write(query)
    response = st.session_state.c_retrieval.invoke(query)["answer"]
    st.chat_message("ai").write(response)
