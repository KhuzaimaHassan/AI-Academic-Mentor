"""
One-time script to build the ChromaDB vector store for RAG system
Loads PDFs from 1_Documentation/RAG_Knowledge_Base/ and creates a persistent vector store
"""

import os
import sys
from pathlib import Path
from typing import List

# Add src code to path for imports
sys.path.append(str(Path(__file__).parent / "4_Src_Code"))

try:
    from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.documents import Document
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required packages: pip install langchain-community pypdf sentence-transformers")
    sys.exit(1)

def build_knowledge_base_vectorstore():
    """
    Build the ChromaDB vector store from PDF documents in RAG_Knowledge_Base folder
    """
    print("🚀 Building RAG Knowledge Base Vector Store...")
    
    # Set up paths
    knowledge_base_path = Path("1_Documentation/RAG_Knowledge_Base")
    vectorstore_path = Path("6_Models/vectorstore")
    
    # Ensure directories exist
    knowledge_base_path.mkdir(parents=True, exist_ok=True)
    vectorstore_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Knowledge base path: {knowledge_base_path}")
    print(f"📁 Vector store path: {vectorstore_path}")
    
    # Check if knowledge base directory has PDFs
    pdf_files = list(knowledge_base_path.glob("*.pdf"))
    print(f"📄 Found {len(pdf_files)} PDF files in knowledge base")
    
    if not pdf_files:
        print("⚠️ No PDF files found in knowledge base directory.")
        print("Please add PDF files to 1_Documentation/RAG_Knowledge_Base/")
        
        # Create a sample document for testing if no PDFs exist
        print("Creating sample knowledge base content...")
        create_sample_knowledge_base(knowledge_base_path)
        pdf_files = list(knowledge_base_path.glob("*.pdf"))
    
    try:
        # Step 1: Load documents from knowledge base directory (PDFs + TXTs)
        print("\n📚 Loading documents from knowledge base (PDFs and TXTs)...")
        documents = []
        
        # Load PDFs if any
        try:
            pdf_loader = PyPDFDirectoryLoader(str(knowledge_base_path))
            pdf_docs = pdf_loader.load()
            documents.extend(pdf_docs)
            print(f"✅ Loaded {len(pdf_docs)} PDF documents")
        except Exception as e:
            print(f"⚠️ Skipping PDF load due to error: {e}")

        # Load .txt fallback documents
        try:
            txt_loader = DirectoryLoader(
                str(knowledge_base_path),
                glob="**/*.txt",
                loader_cls=TextLoader,
                show_progress=True,
            )
            txt_docs = txt_loader.load()
            documents.extend(txt_docs)
            print(f"✅ Loaded {len(txt_docs)} TXT documents")
        except Exception as e:
            print(f"⚠️ Skipping TXT load due to error: {e}")

        print(f"📚 Total documents loaded: {len(documents)}")
        if not documents:
            raise RuntimeError("No documents found to build the vector store.")
        
        # Display document info
        for i, doc in enumerate(documents[:3]):  # Show first 3 docs
            print(f"  Document {i+1}: {len(doc.page_content)} characters")
        
        # Step 2: Split documents into chunks
        print("\n✂️ Splitting documents into chunks...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        splits = text_splitter.split_documents(documents)
        print(f"✅ Created {len(splits)} text chunks")
        
        # Display chunk info
        chunk_sizes = [len(chunk.page_content) for chunk in splits[:5]]
        print(f"Sample chunk sizes: {chunk_sizes}")
        
        # Step 3: Create embedding model
        print("\n🧠 Creating embedding model...")
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        print("✅ Embedding model created")
        
        # Step 4: Create and persist vector store
        print("\n💾 Creating ChromaDB vector store...")
        
        # Remove existing vectorstore if it exists
        if vectorstore_path.exists():
            import shutil
            shutil.rmtree(vectorstore_path)
            print("🗑️ Removed existing vector store")
        
        # Create new vector store
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=str(vectorstore_path)
        )
        
        # Persist the vector store
        vectorstore.persist()
        
        print(f"✅ Vector store created and persisted to {vectorstore_path}")
        
        # Test the vector store
        print("\n🔍 Testing vector store...")
        test_queries = [
            "time management for students",
            "academic procrastination",
            "study strategies",
            "online learning tips"
        ]
        
        for query in test_queries:
            docs = vectorstore.similarity_search(query, k=2)
            print(f"Query: '{query}' -> Found {len(docs)} relevant documents")
            if docs:
                print(f"  First result: {docs[0].page_content[:100]}...")
        
        print("\n🎉 RAG Knowledge Base Vector Store built successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error building vector store: {e}")
        return False

def create_sample_knowledge_base(knowledge_base_path: Path):
    """
    Create sample knowledge base content if no PDFs are available
    """
    print("📝 Creating sample knowledge base content...")
    
    # Sample content about time management and academic success
    sample_content = """
    Time Management for Online Students
    
    1. Create a Study Schedule
    - Allocate specific times for each subject
    - Use digital calendars and reminders
    - Include breaks and buffer time
    
    2. Set Clear Goals
    - Break large tasks into smaller milestones
    - Use SMART goals (Specific, Measurable, Achievable, Relevant, Time-bound)
    - Track progress regularly
    
    3. Eliminate Distractions
    - Find a dedicated study space
    - Turn off notifications during study time
    - Use website blockers if needed
    
    Overcoming Academic Procrastination
    
    1. Identify Triggers
    - Recognize when you're avoiding tasks
    - Understand the underlying reasons
    - Address fear of failure or perfectionism
    
    2. Use the 2-Minute Rule
    - If a task takes less than 2 minutes, do it immediately
    - Break larger tasks into 2-minute chunks
    - Build momentum with small wins
    
    3. Implement the Pomodoro Technique
    - Work for 25 minutes, then take a 5-minute break
    - After 4 pomodoros, take a longer break
    - Track your productivity patterns
    
    Effective Study Strategies
    
    1. Active Learning Techniques
    - Summarize information in your own words
    - Create mind maps and visual aids
    - Teach concepts to others
    
    2. Spaced Repetition
    - Review material at increasing intervals
    - Use flashcards for memorization
    - Mix old and new content
    
    3. Practice Testing
    - Take practice exams regularly
    - Use past papers and sample questions
    - Identify knowledge gaps early
    """
    
    # Create a simple text file that can be treated as a document
    sample_file = knowledge_base_path / "academic_success_guide.txt"
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(sample_content)
    
    print(f"✅ Created sample content: {sample_file}")

def main():
    """Main function to build the vector store"""
    print("=" * 60)
    print("🎓 AI Academic Mentor - RAG Vector Store Builder")
    print("=" * 60)
    
    success = build_knowledge_base_vectorstore()
    
    if success:
        print("\n✅ Vector store building completed successfully!")
        print("You can now use the RAG system for academic mentoring.")
    else:
        print("\n❌ Vector store building failed.")
        print("Please check the error messages above and try again.")
    
    return success

if __name__ == "__main__":
    main()
