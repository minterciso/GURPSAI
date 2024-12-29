import os, getpass
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from gurps_gm_guide_states import State


class TestGURPSGMGuide:
    """A simple class to test an Agent that answers based on the Books we add on the Chroma DB.
    This is to become a full fledged class for the Gurps GM Guide"""
    def __init__(self):
        """The default constructur, it'll load the envionment variables, initialize the LLMs (we use Groq and Ollama) and define a default prompt."""
        load_dotenv(dotenv_path='.env')
        # Check the Environments
        if not os.environ.get("GROQ_API_KEY"):
            os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")
        self.__llm = ChatGroq(model="llama3-8b-8192", temperature=0)
        self.__embeddings = OllamaEmbeddings(model="llama3.2")
        self.__prompt = """
        You are a seasoned GM that is speciallized in helping new GMs on working with GURPS. You must aid the GM by answering it's questions about rules and game dinamics.
        If you don't know the answer to a question, tell the user that you don't know and at least try to pin a range of pages for him to look at.
        Always provide the source book name and pages on your answers. Do not answer anything that wasn't in the original question.
        Tell the source as if it's a wikipedia source.
        Question: {question}
        Context: {context}
        Answer:
        """
        self.__memory = MemorySaver()
    
    def open_chroma_db(self, db_path: str, collection_name: str):
        """Open the ChromaDB that either has the collection we need, or create a new one

        :param db_path: The path to store the ChromaDB
        :type db_path: str
        :param collection_name: The collection name to store at the ChromaDB
        :type collection_name: str
        """
        self.__vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.__embeddings,
            persist_directory=db_path
        )
        
    async def load_pdf(self, fname: str):
        """Read the PDF, split it in chunks and add it on the VectorStore.
        For now, we are utilizing the RecursiveCharacterTextSplitter, but we may use some others as well.

        :param fname: The file name to read the PDF
        :type fname: str
        """
        # Read the PDF
        loader = PyPDFLoader(fname)
        pages = []
        async for page in loader.alazy_load():
            pages.append(page)
        # Split the PDF in chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(pages)
        # Add the PDF to the vectorstore
        _ = self.__vector_store.add_documents(documents=all_splits)
    
    # Graph nodes
    def _retrieve(self, state: State):
        """The node to retrieve the docs based on the similarity with the question."""
        retrieved_docs = self.__vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}
    
    def _generate(self, state: State):
        """The node to generate an answer to the question."""
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = [self.__prompt.format(question=state["question"], context=docs_content)]
        response = self.__llm.invoke(messages)
        return {"answer": response.content}
    
    def create_graph(self):
        """Create the simple graph to search our RAG and provide an answer."""
        self.__config = {'configurable': {'thread_id': 'TestGURPSGMGuide3'}}
        
        graph_builder = StateGraph(State)
        graph_builder.add_node("retrieve", self._retrieve)
        graph_builder.add_node("generate", self._generate)
        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        graph_builder.add_edge("generate", END)
        self.__graph = graph_builder.compile(checkpointer=self.__memory)
    
    def run(self):
        """Execute a 'CLI' to talk to the agent"""
        while True:
            user_input = input('> ')
            if user_input.lower() == 'exit':
                print('Goodbye!')
                break
            response = self.__graph.invoke({'question': user_input}, self.__config)
            print(f'GM> {response["answer"]}')
        