import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever
from sentence_transformers import SentenceTransformer, util
from langchain_core.chat_history import BaseChatMessageHistory
import faiss
import shutil
import os

# Templates HTML para mensajes
bot_template = '''
<div style="display: flex; align-items: center; margin-bottom: 10px;">
    <div style="flex-shrink: 0; margin-right: 10px;">
        <img src="https://uxwing.com/wp-content/themes/uxwing/download/communication-chat-call/answer-icon.png" 
             style="max-height: 50px; max-width: 50px; border-radius: 50%; object-fit: cover;">
    </div>
    <div style="background-color: #f1f1f1; padding: 10px; border-radius: 10px; max-width: 75%; word-wrap: break-word; overflow-wrap: break-word;">
        {msg}
    </div>
</div>
'''

user_template = '''
<div style="display: flex; align-items: center; margin-bottom: 10px; justify-content: flex-end;">
    <div style="flex-shrink: 0; margin-left: 10px;">
        <img src="https://cdn.iconscout.com/icon/free/png-512/free-q-characters-character-alphabet-letter-36051.png?f=webp&w=512" 
             style="max-height: 50px; max-width: 50px; border-radius: 50%; object-fit: cover;">
    </div>    
    <div style="background-color: #007bff; color: white; padding: 10px; border-radius: 10px; max-width: 75%; word-wrap: break-word; overflow-wrap: break-word;">
        {msg}
    </div>
</div>
'''

button_style = """
<style>
    .small-button {
        display: inline-block;
        padding: 5px 10px;
        font-size: 12px;
        color: white;
        background-color: #007bff;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        margin-right: 5px;
    }
    .small-button:hover {
        background-color: #0056b3;
    }
</style>
"""

# Función para limpiar la base de datos FAISS
def clear_vector_db():
    try:
        os.remove("faiss_index")  # Elimina el archivo de índice FAISS
        st.sidebar.success("La base de datos vectorial ha sido limpiada.")
    except FileNotFoundError:
        st.sidebar.warning("La base de datos ya está vacía.")

# Función para preparar y dividir documentos
def prepare_and_split_docs(pdf_files):
    split_docs = []
    for pdf in pdf_files:
        with open(pdf.name, "wb") as f:
            f.write(pdf.getbuffer())

        loader = PyPDFLoader(pdf.name)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=512, chunk_overlap=256
        )
        split_docs.extend(splitter.split_documents(documents))
    return split_docs

# Función para ingerir documentos en FAISS usando HuggingFaceEmbeddings
def ingest_into_vectordb(split_docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(split_docs, embeddings)
    faiss.write_index(db.index, "faiss_index")  # Guardar el índice
    return db

# Configuración del LLM Groq y cadena de recuperación
def get_conversation_chain(retriever):
    llm = ChatGroq(model="mixtral-8x7b-32768", api_key="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")
    contextualize_q_system_prompt = (
        "Given the chat history and the latest user question, "
        "provide a response that directly addresses the user's query based on the provided documents."
    
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    system_prompt = (
        "As a personal chat assistant, provide accurate and relevant information based on the provided document in 2-3 sentences. "
        "Answer should be limited to 50 words and 2-3 sentences. Do not prompt to select answers or formulate standalone questions. Do not ask questions in the response. "
        "Do not respond if the query does not find the documents provided"
        "respond in Spanish"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain

st.title("Consulta tus documentos aquí :books:")

# Cargar documentos PDF
uploaded_files = st.sidebar.file_uploader("Sube documentos PDF", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.sidebar.button("Procesar PDFs"):
    split_docs = prepare_and_split_docs(uploaded_files)
    vector_db = ingest_into_vectordb(split_docs)
    retriever = vector_db.as_retriever()
    st.sidebar.success("Documentos procesados y base de datos creada!")

    conversational_chain = get_conversation_chain(retriever)
    st.session_state.conversational_chain = conversational_chain

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
#
if 'show_docs' not in st.session_state:
    st.session_state.show_docs = {}


def toggle_docs(index):
    st.session_state.show_docs[index] = not st.session_state.show_docs.get(index, False)
#
user_input = st.text_input("Haz una pregunta sobre los documentos:")

if st.button("Enviar"):
    st.markdown(button_style, unsafe_allow_html=True)
    if user_input and 'conversational_chain' in st.session_state:
        session_id = "session1"
        conversational_chain = st.session_state.conversational_chain
        response = conversational_chain.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})
        #st.session_state.chat_history.append({"user": user_input, "bot": response['answer']})
        ##
        context_docs = response.get('context', [])
        st.session_state.chat_history.append({"user": user_input, "bot": response['answer'], "context_docs": context_docs})
        ##
if st.session_state.chat_history:
    for index,message in enumerate(st.session_state.chat_history):
        st.markdown(user_template.format(msg=message['user']), unsafe_allow_html=True)
        st.markdown(bot_template.format(msg=message['bot']), unsafe_allow_html=True)
#
        if f"show_docs_{index}" not in st.session_state:
            st.session_state[f"show_docs_{index}"] = False
        if f"similarity_score_{index}" not in st.session_state:
            st.session_state[f"similarity_score_{index}"] = None

        cols = st.columns([1, 1])

        with cols[0]:
            if st.button(f"Mostrar/Ocultar Recursos Docs", key=f"toggle_{index}"):
                st.session_state[f"show_docs_{index}"] = not st.session_state[f"show_docs_{index}"]


        if st.session_state[f"show_docs_{index}"]:
            with st.expander("Extractos del documento"):
                for doc in message.get('context_docs', []):
                    st.write(f"Source: {doc.metadata['source']}")
                    st.write(doc.page_content)

        
#



if st.sidebar.button("Limpiar base de datos"):
    clear_vector_db()
