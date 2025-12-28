"""
FAISS Vector Store Manager

This module handles FAISS vector store initialization and management.
It loads example queries and extra prompt data into FAISS for semantic search.
"""

import os
from typing import List, Dict
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv
from examples_data import SAMPLE_EXAMPLES, EXTRA_PROMPT_DATA

# Load environment variables
load_dotenv()

# Path to store FAISS index
FAISS_INDEX_PATH = "./faiss_index"
EXAMPLES_INDEX_PATH = os.path.join(FAISS_INDEX_PATH, "examples")
EXTRA_PROMPT_INDEX_PATH = os.path.join(FAISS_INDEX_PATH, "extra_prompts")


class VectorStoreManager:
    """
    Manages FAISS vector stores for examples and extra prompt data.
    """
    
    def __init__(self):
        """Initialize embeddings model."""
        # Get API key from environment
        api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API_KEY or OPENAI_API_KEY must be set in environment variables")
        
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.examples_store = None
        self.extra_prompts_store = None
    
    def _create_documents_from_examples(self) -> List[Document]:
        """Convert example data to Document objects."""
        documents = []
        for example in SAMPLE_EXAMPLES:
            # Combine question and SQL for better semantic search
            content = f"Question: {example['question']}\n\nSQL Query:\n{example['sql']}"
            doc = Document(
                page_content=content,
                metadata=example.get("metadata", {})
            )
            documents.append(doc)
        return documents
    
    def _create_documents_from_extra_prompts(self) -> List[Document]:
        """Convert extra prompt data to Document objects."""
        documents = []
        for item in EXTRA_PROMPT_DATA:
            doc = Document(
                page_content=item["content"],
                metadata=item.get("metadata", {})
            )
            documents.append(doc)
        return documents
    
    def _index_exists(self, index_path: str) -> bool:
        """Check if FAISS index already exists."""
        return os.path.exists(index_path) and os.path.isdir(index_path)
    
    def initialize_stores(self):
        """
        Initialize FAISS vector stores.
        Creates indexes if they don't exist, otherwise loads existing ones.
        """
        # Create directory if it doesn't exist
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        
        # Initialize examples store
        if self._index_exists(EXAMPLES_INDEX_PATH):
            print(f"Loading existing FAISS index from {EXAMPLES_INDEX_PATH}")
            self.examples_store = FAISS.load_local(
                EXAMPLES_INDEX_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print(f"Creating new FAISS index for examples at {EXAMPLES_INDEX_PATH}")
            example_docs = self._create_documents_from_examples()
            self.examples_store = FAISS.from_documents(example_docs, self.embeddings)
            self.examples_store.save_local(EXAMPLES_INDEX_PATH)
            print(f"✅ Examples vector store created with {len(example_docs)} documents")
        
        # Initialize extra prompts store
        if self._index_exists(EXTRA_PROMPT_INDEX_PATH):
            print(f"Loading existing FAISS index from {EXTRA_PROMPT_INDEX_PATH}")
            self.extra_prompts_store = FAISS.load_local(
                EXTRA_PROMPT_INDEX_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print(f"Creating new FAISS index for extra prompts at {EXTRA_PROMPT_INDEX_PATH}")
            extra_docs = self._create_documents_from_extra_prompts()
            self.extra_prompts_store = FAISS.from_documents(extra_docs, self.embeddings)
            self.extra_prompts_store.save_local(EXTRA_PROMPT_INDEX_PATH)
            print(f"✅ Extra prompts vector store created with {len(extra_docs)} documents")
    
    def search_examples(self, query: str, k: int = 3) -> List[Document]:
        """
        Search for similar examples in the vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar example documents
        """
        if not self.examples_store:
            print(f"⚠️  Examples store not initialized")
            return []
        
        print(f"\n{'='*80}")
        print(f"🔍 VECTOR STORE SEARCH - Examples")
        print(f"{'='*80}")
        print(f"   Query: {query}")
        print(f"   K: {k}")
        
        retriever = self.examples_store.as_retriever(search_kwargs={"k": k})
        results = retriever.invoke(query)
        
        print(f"   Found {len(results)} results:")
        for i, doc in enumerate(results, 1):
            content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            print(f"   [{i}] {content_preview}")
            if doc.metadata:
                print(f"       Metadata: {doc.metadata}")
        print(f"{'='*80}\n")
        
        return results
    
    def search_extra_prompts(self, query: str, k: int = 2) -> List[Document]:
        """
        Search for relevant extra prompt data.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant prompt documents
        """
        if not self.extra_prompts_store:
            print(f"⚠️  Extra prompts store not initialized")
            return []
        
        print(f"\n{'='*80}")
        print(f"🔍 VECTOR STORE SEARCH - Extra Prompts")
        print(f"{'='*80}")
        print(f"   Query: {query}")
        print(f"   K: {k}")
        
        retriever = self.extra_prompts_store.as_retriever(search_kwargs={"k": k})
        results = retriever.invoke(query)
        
        print(f"   Found {len(results)} results:")
        for i, doc in enumerate(results, 1):
            content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            print(f"   [{i}] {content_preview}")
            if doc.metadata:
                print(f"       Metadata: {doc.metadata}")
        print(f"{'='*80}\n")
        
        return results

