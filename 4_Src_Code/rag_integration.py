"""
RAG Integration Module for AI Academic Mentor
Implements Retrieval-Augmented Generation using ChromaDB vector store
"""

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import json
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pandas as pd
from pathlib import Path

class RAGIntegration:
    """Implements RAG system using ChromaDB for academic knowledge base"""
    
    def __init__(self, persist_directory: str = "6_Models/vectorstore"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.vectorstore = None
        self.knowledge_base_path = "1_Documentation/RAG_Knowledge_Base"
        
        # Ensure directories exist
        os.makedirs(persist_directory, exist_ok=True)
        os.makedirs(self.knowledge_base_path, exist_ok=True)
    
    def create_knowledge_base_from_data(self, datasets: Dict[str, pd.DataFrame]) -> None:
        """Create knowledge base from OULAD dataset"""
        
        documents = []
        
        # Create documents from each dataset
        for dataset_name, df in datasets.items():
            # Create summary document for the dataset
            summary_text = f"""
            Dataset: {dataset_name}
            Shape: {df.shape}
            Columns: {list(df.columns)}
            
            Description: This dataset contains information about {dataset_name.replace('_', ' ')}.
            """
            
            documents.append(Document(
                page_content=summary_text,
                metadata={"source": dataset_name, "type": "dataset_summary"}
            ))
            
            # Create documents for key insights
            if dataset_name == "studentInfo":
                insights = self._extract_student_insights(df)
                for insight in insights:
                    documents.append(Document(
                        page_content=insight,
                        metadata={"source": dataset_name, "type": "insight"}
                    ))
            
            elif dataset_name == "studentAssessment":
                insights = self._extract_assessment_insights(df)
                for insight in insights:
                    documents.append(Document(
                        page_content=insight,
                        metadata={"source": dataset_name, "type": "assessment_insight"}
                    ))
        
        # Create vector store
        self._create_vectorstore(documents)
        
        print(f"Created knowledge base with {len(documents)} documents")
    
    def _extract_student_insights(self, df: pd.DataFrame) -> List[str]:
        """Extract insights from student info dataset"""
        insights = []
        
        # Performance by gender
        if 'gender' in df.columns and 'final_result' in df.columns:
            gender_performance = df.groupby('gender')['final_result'].value_counts()
            insights.append(f"Gender performance patterns: {gender_performance.to_dict()}")
        
        # Performance by region
        if 'region' in df.columns and 'final_result' in df.columns:
            region_performance = df.groupby('region')['final_result'].value_counts()
            insights.append(f"Regional performance patterns: {region_performance.to_dict()}")
        
        # Education level impact
        if 'highest_education' in df.columns and 'final_result' in df.columns:
            education_performance = df.groupby('highest_education')['final_result'].value_counts()
            insights.append(f"Education level impact: {education_performance.to_dict()}")
        
        return insights
    
    def _extract_assessment_insights(self, df: pd.DataFrame) -> List[str]:
        """Extract insights from assessment dataset"""
        insights = []
        
        # Score distribution
        if 'score' in df.columns:
            score_stats = df['score'].describe()
            insights.append(f"Assessment score statistics: Mean={score_stats['mean']:.2f}, Std={score_stats['std']:.2f}")
        
        # Assessment patterns
        if 'id_assessment' in df.columns:
            assessment_counts = df['id_assessment'].value_counts()
            insights.append(f"Most common assessments: {assessment_counts.head(5).to_dict()}")
        
        return insights
    
    def _create_vectorstore(self, documents: List[Document]) -> None:
        """Create and persist vector store"""
        
        # Split documents
        texts = self.text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Persist the vector store
        self.vectorstore.persist()
        
        print(f"Vector store created and persisted to {self.persist_directory}")
    
    def load_vectorstore(self) -> bool:
        """Load existing vector store"""
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print("Vector store loaded successfully")
            return True
        except Exception as e:
            print(f"Failed to load vector store: {e}")
            return False
    
    def add_documents_to_knowledge_base(self, documents: List[Document]) -> None:
        """Add new documents to the knowledge base"""
        if self.vectorstore is None:
            self.load_vectorstore()
        
        if self.vectorstore is None:
            print("No vector store available. Please create one first.")
            return
        
        # Split documents
        texts = self.text_splitter.split_documents(documents)
        
        # Add to vector store
        self.vectorstore.add_documents(texts)
        self.vectorstore.persist()
        
        print(f"Added {len(texts)} documents to knowledge base")
    
    def retrieve_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents for a query"""
        if self.vectorstore is None:
            print("No vector store available. Please create one first.")
            return []
        
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return docs
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def get_context_for_query(self, query: str, k: int = 5) -> str:
        """Get relevant context for a query"""
        documents = self.retrieve_relevant_documents(query, k)
        
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"Context {i}: {doc.page_content}")
        
        return "\n\n".join(context_parts)
    
    def create_academic_recommendations_context(self, student_features: Dict[str, Any]) -> str:
        """Create context for academic recommendations"""
        
        # Build query based on student features
        query_parts = []
        
        if 'performance_level' in student_features:
            query_parts.append(f"student performance {student_features['performance_level']}")
        
        if 'risk_factors' in student_features:
            query_parts.append(f"academic risk factors {student_features['risk_factors']}")
        
        if 'engagement_score' in student_features:
            query_parts.append(f"student engagement {student_features['engagement_score']}")
        
        query = " ".join(query_parts) if query_parts else "academic performance improvement strategies"
        
        return self.get_context_for_query(query, k=3)
    
    def search_knowledge_base(self, query: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search knowledge base with optional filters"""
        if self.vectorstore is None:
            print("No vector store available. Please create one first.")
            return []
        
        try:
            # Basic similarity search
            docs = self.vectorstore.similarity_search(query, k=10)
            
            results = []
            for doc in docs:
                result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": None  # ChromaDB doesn't return scores by default
                }
                results.append(result)
            
            return results
        except Exception as e:
            print(f"Error searching knowledge base: {e}")
            return []
    
    def update_knowledge_base_with_results(self, analysis_results: Dict[str, Any]) -> None:
        """Update knowledge base with new analysis results"""
        
        # Create document from analysis results
        analysis_text = f"""
        Academic Analysis Results:
        Student Analysis: {analysis_results.get('student_analysis', '')}
        Key Insights: {analysis_results.get('key_insights', '')}
        Recommendations: {analysis_results.get('personalized_recommendations', '')}
        Timestamp: {analysis_results.get('timestamp', '')}
        """
        
        document = Document(
            page_content=analysis_text,
            metadata={
                "source": "analysis_results",
                "type": "mentorship_session",
                "timestamp": analysis_results.get('timestamp', '')
            }
        )
        
        self.add_documents_to_knowledge_base([document])
    
    def get_rag_retriever(self):
        """
        Loads the persistent ChromaDB from 6_Models/vectorstore/ and returns a retriever object
        
        Returns:
            ChromaDB retriever object or None if loading fails
        """
        try:
            # Load the persistent vector store
            if self.vectorstore is None:
                success = self.load_vectorstore()
                if not success:
                    print("❌ Failed to load vector store for retriever")
                    return None
            
            # Create retriever from the vector store
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            print("✅ RAG retriever created successfully")
            return retriever
            
        except Exception as e:
            print(f"❌ Error creating RAG retriever: {e}")
            return None
    
    def get_intervention_advice(self, query: str) -> str:
        """
        Gets intervention advice by querying the RAG knowledge base
        
        Args:
            query: The query string for academic intervention advice
            
        Returns:
            Combined string of retrieved documents' content
        """
        try:
            print(f"🔍 Getting intervention advice for query: '{query}'")
            
            # Get the retriever
            retriever = self.get_rag_retriever()
            
            if retriever is None:
                return "❌ Unable to access knowledge base. Please ensure the vector store is properly initialized."
            
            # Retrieve relevant documents
            retrieved_docs = retriever.invoke(query)
            
            if not retrieved_docs:
                return "No relevant intervention advice found in the knowledge base."
            
            print(f"✅ Retrieved {len(retrieved_docs)} relevant documents")
            
            # Combine document contents
            combined_content = []
            for i, doc in enumerate(retrieved_docs, 1):
                content_snippet = doc.page_content.strip()
                # Add source information if available
                source_info = doc.metadata.get('source', 'Unknown source')
                combined_content.append(f"Source {i} ({source_info}):\n{content_snippet}")
            
            # Join all content with separators
            final_content = "\n\n" + "="*50 + "\n\n".join(combined_content)
            
            print(f"✅ Generated intervention advice with {len(final_content)} characters")
            return final_content
            
        except Exception as e:
            error_msg = f"❌ Error getting intervention advice: {e}"
            print(error_msg)
            return error_msg
