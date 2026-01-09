import os
import time
import tempfile
import streamlit as st 
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader # Document Loader 
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter   # Doucment Text Splitter 
from langchain_huggingface import HuggingFaceEmbeddings  # Doucment  Embedding 
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.runnables import  RunnableWithMessageHistory



def loadPdfFromFilepicker(uploaded_file):
    ''' This funtion takes Streamlit file picker output and returns the loaded PDF 
    docuement to be used in Langchain'''      
    
    # Use a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name
    try:
        # Initialize PyPDFLoader with the temporary file path
        loader = PyPDFLoader(temp_file_path)
        
        # Load the documents
        # This will extract text page by page, including metadata like page numbers
        documents = loader.load()
        # successmessage = st.success("File uploaded successfully!")
        pageloadinfo =st.info(f"Total pages loaded: {len(documents)}")
        
    except Exception as e:
        st.error(f"An error occurred during PDF processing: {e}")          
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
    return documents



st.header('RAG Example - Document QnA')
st.image(r'D:\AI_ML\AI_LangChain\KrishNiakk\Apps\2_RAG_GenAI\RAGBG.png')

with st.sidebar:
    st.session_state.session_id = st.text_input('Enter your name to create a new session:').replace(' ','')
    uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")
    if uploaded_file is not None:
        pdfData = loadPdfFromFilepicker(uploaded_file)
    
    model_name = st.selectbox('Choose Model', options= ['openai/gpt-oss-120b',
                                                        'llama-3.1-8b-instant',
                                                        'llama-3.3-70b-versatile',
                                                        'groq/compound',
                                                        'openai/gpt-oss-120b'] )
    api_key = st.text_input('Paste your GROQ Key:', type= 'password')
    temprature = st.slider(label='Creativity',  min_value= 0.2 , max_value=1.0,value= 0.7 )

    if st.session_state.session_id != '' :
        # session_id = session_id+str(int(time.time()))
        st.text(f'Session_id : {st.session_state.session_id}')
    else:
        st.session_state.session_id ='default'

    if model_name:
        st.text(model_name)

if "chat_history" not in st.session_state :
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:
    with st.chat_message(chat['role']):
        st.markdown(chat['content'])

if "store" not in st.session_state:
    st.session_state.store = {}
store = st.session_state.store 
def get_session_history( session_id : str) -> BaseChatMessageHistory:
    if session_id not in store :
        store[session_id] = ChatMessageHistory()
    return  store[session_id]

config = {'configurable' : {'session_id': st.session_state.session_id}}

if (uploaded_file is not None) and  (api_key != ''):
    with st.expander("Document Preview"):
            for i in range(min(2, len(pdfData))):
                st.write(pdfData[i].page_content)

    ## Chat Client
    client = ChatGroq(model=model_name, groq_api_key= api_key, temperature=temprature)
    ## Document Splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size =1000, chunk_overlap = 200)
    splittedDocs = splitter.split_documents(pdfData)

    ### Setting up Embedding Engine  
    embedding = HuggingFaceEmbeddings(model= 'sentence-transformers/all-MiniLM-L6-v2')

    ## Vectore Stores and Base Retriever
    if 'vectorDB' not in st.session_state:
        st.session_state.vectorDB = Chroma.from_documents(
            documents= splittedDocs, 
            embedding= embedding,
            persist_directory="./vector_db")
    retreiver = st.session_state.vectorDB.as_retriever(search_kwargs={"k": 6})
    ## Base Prompt
    systemPrompt  = '''You are a helpful polite and  assistant. 
                        and 
                        answer all the query politely based on the context provided. 
                        If you dont find the relevent answer in the context repond respectfully 
                        that you don't now the correct answer due to limited access and 
                        can refer only the attached docuemtn
                          
                        Use the the context and create descriptive answer.
                        At the end of the response always  ask politely what user
                        want to know or search from the provided document.
                        You need not to search answer anywhere apart from the given context
                        \n\n
                        <context>
                        {context}
                        <context>
                        '''

    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', systemPrompt),
            ('human' , "{input}")
        ] )

    ## Basic RAG Chain
    question_answer_chian = create_stuff_documents_chain(client, prompt)
    rag_chain = create_retrieval_chain(retriever=retreiver, combine_docs_chain= question_answer_chian)

    # response  = rag_chain.invoke(
    #     {"input" :   'What are the different  Spatial libraries we have'
    #     })

    contextulize_q_system_prompt = ''' Given a chat history and latest user question and
                                    which might reference the context in the chat history,
                                    AI raised question and responses ,
                                    formulate a standalone question which can be understood 
                                    without the chat history. 
                                    Do not answer the question, 
                                    just reformulate it if needed and otherwise return it as is. '''

    contextulize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ('system' ,contextulize_q_system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human' , '{input}')
        ]
    )

    historyAwareRetreiver = create_history_aware_retriever(llm= client,retriever=  retreiver ,prompt=contextulize_q_prompt)


    qaPrompt = ChatPromptTemplate.from_messages(
        [
            ('system' ,systemPrompt),
            MessagesPlaceholder(variable_name= 'chat_history'),
            ('human' , '{input}')
        ]
    )

    qaChain = create_stuff_documents_chain(client, qaPrompt)
    historyRagChain = create_retrieval_chain(historyAwareRetreiver, qaChain)


    withMessageHistory = RunnableWithMessageHistory( historyRagChain,
                                                    get_session_history=get_session_history,
                                                    history_messages_key= 'chat_history',
                                                    input_messages_key= 'input',
                                                    output_messages_key= 'answer'
                                                        )



    userQuery = st.chat_input(placeholder= 'Query Your Document ', max_chars = 200)


    if userQuery :
        with st.chat_message(name= 'human'):
            st.markdown(userQuery)
        st.session_state.chat_history.append({"role": 'human', "content" :userQuery})
        
        response1 = withMessageHistory.invoke(
            {'input': userQuery},
            config = config
                )
        with st.expander('Retrived Context'):
            for doc in response1['context']:
                st.write(doc.page_content)
                print(doc.page_content)
                    
            #         st.messages(name= "🧑‍💻"response1['context'])
            
        with st.chat_message(name = 'ai'):
            st.markdown(response1['answer'])
        
        st.session_state.chat_history.append({"role": 'ai', "content" :response1['answer']})

