"""
title: SberAI RAG Pipeline for OpenWebUI
author: butterboard
version: 1.1.0
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Union, Generator, Iterator, Optional
from dotenv import load_dotenv
import pathlib

from operator import itemgetter

# LangChain components
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# SberAI models
from langchain_community.embeddings import GigaChatEmbeddings
from langchain_community.llms import GigaChat

def get_user_progress(id):
    return 3

class Pipeline:
    def __init__(self):
        self.vector_store = None
        self.embeddings = None
        self.llm = None
        self.rag_chain = None
        self.persist_dir = "./chroma_db"
        self.docs_dir = "./docs"
        self.initialized = False

    async def on_startup(self):
        """Initialize pipeline components"""
        load_dotenv()  # Load environment variables
        
        # Initialize SberAI models
        self.embeddings = GigaChatEmbeddings(
            credentials=os.getenv("GIGACHAT_API_KEY"),
            scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
            verify_ssl_certs=False
        )

        self.llm = GigaChat(
            credentials=os.getenv("GIGACHAT_API_KEY"),
            scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
            model=os.getenv("GIGACHAT_MODEL", "GigaChat"),
            verify_ssl_certs=False,
            temperature=0.7
        )

        # Create document directory if missing
        Path(self.docs_dir).mkdir(parents=True, exist_ok=True)

        # Initialize or load vector store
        if Path(self.persist_dir).exists():
            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
        else:
            await self._process_documents()
        
        # Build RAG chain
        self.retriever = self.vector_store.as_retriever()
        
        self.user_progress_db = {}  # In-memory storage; replace with real DB in production
        
        # Define the progress tracking function
        self.get_user_progress = lambda user_id: self.user_progress_db.get(user_id, 0)
        
        # Set up the RAG pipeline with integrated system prompt
        self.prompt = ChatPromptTemplate.from_template(
        """Ты — дружелюбный ассистент онбординга, помогающий новым сотрудникам. Обращайся по имени, если известно.
            Сохраняй теплый, поддерживающий тон. Напоминай, что ты в стадии тестирования.
            
            Правила:
            1. Отвечай ТОЛЬКО на основе контекста (база знаний компании):
            {context}
            
            2. Всегда предлагай задать дополнительные вопросы по онбордингу
            
            3. Рекомендуй статьи из корпоративной вики
            
            4. Учитывай прогресс адаптации (завершено шагов: {progress}/5)
            
            5. Напоминай: "Я еще учусь - пожалуйста, перепроверяй важную информацию в HR"
            
            Вопрос ({language}): {question}
            
            Отвечай на {language} в формате:
            [Приветствие с именем] [Ответ по контексту]
            [Предложение задать следующий вопрос] [Прогресс, если уместно]
            [Напоминание о тестировании]"""
        )
        
        self.rag_chain = (
            {
                "context": self.retriever,
                "question": itemgetter("question"),
                "language": itemgetter("language"),
                "progress": lambda x: str(self.get_user_progress(x["user_id"]))
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        self.initialized = True
    
    def update_progress(self, user_id, new_step):
        """Update user's progress in the onboarding process"""
        current = self.user_progress_db.get(user_id, 0)
        if new_step > current:  # Only move forward, not backward
            self.user_progress_db[user_id] = new_step
        return self.user_progress_db[user_id]
    
    def generate_response(self, user_input):
        """Process user query with progress tracking"""
        response = self.rag_chain.invoke({
            "question": user_input["question"],
            "language": user_input.get("language", "en"),  # Default to English
            "user_id": user_input["user_id"]
        })
        
        # Optional: Auto-update progress if certain keywords are detected
        if "completed" in user_input["question"].lower():
            self.update_progress(user_input["user_id"], 
                               self.get_user_progress(user_input["user_id"]) + 1)
        
        return response


    async def on_shutdown(self):
        """Cleanup resources"""
        if self.vector_store:
            self.vector_store.persist()
        self.initialized = False

    async def _process_documents(self):
        """Process and store markdown documents"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        documents = []
        md_files = list(Path(self.docs_dir).glob("**/*.md"))
        
        if not md_files:
            raise ValueError(f"No markdown files found in {self.docs_dir}")

        for md_file in md_files:
            try:
                loader = UnstructuredMarkdownLoader(str(md_file))
                docs = loader.load()
                split_docs = text_splitter.split_documents(docs)
                documents.extend(split_docs)
            except Exception as e:
                print(f"Error processing {md_file}: {str(e)}")
                continue

        if not documents:
            raise ValueError("No valid documents processed")

        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        self.vector_store.persist()

    def pipe(
        self, 
        user_message: str, 
        model_id: str, 
        messages: List[Dict], 
        body: Dict
    ) -> Union[str, Generator, Iterator]:
        """Main processing pipeline"""
        
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized - call on_startup first")

        # Create response generator
        def response_generator():
            try:
                # Get last user message
                query = messages[-1]["content"] if messages else user_message
                
                # Stream response
                for chunk in self.rag_chain.stream(query):
                    yield chunk
                    
            except Exception as e:
                yield f"Error: {str(e)}"

        return response_generator()