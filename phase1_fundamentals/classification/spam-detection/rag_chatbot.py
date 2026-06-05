"""
Simple RAG (Retrieval Augmented Generation) Chatbot
Uses Claude/OpenAI API + Vector Database for context-aware responses

This demonstrates modern Generative AI concepts:
1. Document embedding & storage
2. Semantic search with vector similarity
3. LLM integration for generation
4. Context augmentation (RAG)
"""

import os
from typing import List, Dict, Optional
import json
from pathlib import Path

# Check if required packages are installed
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    print("⚠️  Anthropic SDK not installed. Install with: pip install anthropic")

try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  scikit-learn not installed")

# ============== SIMPLE VECTOR STORE ==============

class SimpleVectorStore:
    """
    Simple in-memory vector store using TF-IDF
    In production, use: Pinecone, Weaviate, ChromaDB, or FAISS
    """

    def __init__(self):
        self.documents: List[Dict] = []
        self.vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
        self.vectors = None

    def add_documents(self, docs: List[Dict]):
        """
        Add documents to the store
        docs = [{'id': str, 'text': str, 'metadata': dict}, ...]
        """
        self.documents.extend(docs)

        # Extract text and create vectors
        texts = [doc['text'] for doc in self.documents]
        self.vectors = self.vectorizer.fit_transform(texts).toarray()

        print(f"✅ Added {len(docs)} documents. Total: {len(self.documents)}")

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search for similar documents using cosine similarity
        """
        if not self.documents:
            return []

        # Vectorize query
        query_vector = self.vectorizer.transform([query]).toarray()

        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.vectors)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return documents with scores
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'score': float(similarities[idx])
            })

        return results

# ============== RAG CHATBOT ==============

class RAGChatbot:
    """
    RAG Chatbot using Claude/OpenAI for generation
    """

    def __init__(self, api_key: Optional[str] = None, provider: str = "anthropic"):
        """
        Initialize chatbot
        provider: 'anthropic' or 'openai'
        """
        self.provider = provider
        self.vector_store = SimpleVectorStore()
        self.conversation_history: List[Dict] = []

        if provider == "anthropic":
            if not HAS_ANTHROPIC:
                raise ImportError("Install anthropic: pip install anthropic")
            self.client = anthropic.Anthropic(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
            )
            self.model = "claude-3-5-sonnet-20241022"
        else:
            raise ValueError(f"Provider {provider} not supported yet")

    def load_knowledge_base(self, documents: List[Dict]):
        """
        Load documents into vector store (knowledge base)
        """
        self.vector_store.add_documents(documents)

    def load_from_text_file(self, file_path: str, chunk_size: int = 500):
        """
        Load knowledge from a text file by splitting into chunks
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Simple chunking (split by paragraphs or sentences)
        chunks = content.split('\n\n')  # Split by double newline

        documents = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                documents.append({
                    'id': f'chunk_{i}',
                    'text': chunk.strip(),
                    'metadata': {'source': file_path, 'chunk_id': i}
                })

        self.load_knowledge_base(documents)

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve relevant context from vector store
        """
        results = self.vector_store.search(query, top_k=top_k)

        if not results:
            return ""

        # Format context
        context_parts = []
        for i, result in enumerate(results, 1):
            score = result['score']
            text = result['document']['text']
            context_parts.append(f"[Context {i} - Relevance: {score:.2f}]\n{text}")

        return "\n\n".join(context_parts)

    def chat(self, user_message: str, use_rag: bool = True) -> str:
        """
        Chat with the bot (with or without RAG)
        """
        # Retrieve context if RAG is enabled
        context = ""
        if use_rag and self.vector_store.documents:
            context = self.retrieve_context(user_message, top_k=3)

        # Build system prompt
        system_prompt = "You are a helpful AI assistant."
        if context:
            system_prompt += f"\n\nYou have access to the following context information:\n\n{context}\n\n"
            system_prompt += "Use this context to answer the user's question accurately. If the context doesn't contain relevant information, say so."

        # Add to conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': user_message
        })

        # Call LLM
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=self.conversation_history
            )
            assistant_message = response.content[0].text

        # Add response to history
        self.conversation_history.append({
            'role': 'assistant',
            'content': assistant_message
        })

        return assistant_message

    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []

# ============== DEMO & EXAMPLES ==============

def create_ml_knowledge_base():
    """
    Create a sample knowledge base about ML concepts
    """
    documents = [
        {
            'id': 'doc_1',
            'text': 'Naive Bayes is a probabilistic classifier based on Bayes theorem. It assumes independence between features. It works well for text classification tasks like spam detection.',
            'metadata': {'category': 'algorithms', 'difficulty': 'beginner'}
        },
        {
            'id': 'doc_2',
            'text': 'Random Forest is an ensemble learning method that constructs multiple decision trees during training. It outputs the mode of predictions from individual trees. It handles non-linear patterns well.',
            'metadata': {'category': 'algorithms', 'difficulty': 'intermediate'}
        },
        {
            'id': 'doc_3',
            'text': 'Logistic Regression is a linear model for binary classification. It uses a sigmoid function to map predictions to probabilities between 0 and 1. The weights can be interpreted directly.',
            'metadata': {'category': 'algorithms', 'difficulty': 'beginner'}
        },
        {
            'id': 'doc_4',
            'text': 'Cross-validation is a technique to assess model generalization. K-fold CV splits data into k subsets, trains on k-1 folds and validates on the remaining fold. This is repeated k times.',
            'metadata': {'category': 'evaluation', 'difficulty': 'intermediate'}
        },
        {
            'id': 'doc_5',
            'text': 'TF-IDF (Term Frequency-Inverse Document Frequency) is a feature extraction technique for text. It measures word importance by considering both local frequency and global rarity. Higher values indicate more distinctive terms.',
            'metadata': {'category': 'feature_engineering', 'difficulty': 'intermediate'}
        },
        {
            'id': 'doc_6',
            'text': 'Precision measures the proportion of positive predictions that are actually correct (TP / (TP + FP)). Recall measures the proportion of actual positives that were correctly identified (TP / (TP + FN)).',
            'metadata': {'category': 'evaluation', 'difficulty': 'intermediate'}
        },
        {
            'id': 'doc_7',
            'text': 'RAG (Retrieval Augmented Generation) combines information retrieval with language generation. It retrieves relevant documents from a knowledge base and uses them as context for the LLM to generate accurate, grounded responses.',
            'metadata': {'category': 'nlp', 'difficulty': 'advanced'}
        }
    ]
    return documents

def demo_without_api():
    """Demo vector search without LLM API"""
    print("="*70)
    print("🔍 DEMO: Vector Search (No LLM API Required)")
    print("="*70)

    # Create vector store
    store = SimpleVectorStore()
    docs = create_ml_knowledge_base()
    store.add_documents(docs)

    # Test queries
    queries = [
        "What is Naive Bayes?",
        "How does cross-validation work?",
        "Explain precision and recall"
    ]

    for query in queries:
        print(f"\n📝 Query: {query}")
        results = store.search(query, top_k=2)

        for i, result in enumerate(results, 1):
            print(f"\n  Result {i} (Score: {result['score']:.3f}):")
            print(f"  {result['document']['text'][:150]}...")

def demo_with_api():
    """Demo full RAG chatbot with LLM"""
    print("\n" + "="*70)
    print("🤖 DEMO: RAG Chatbot with Claude")
    print("="*70)

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n⚠️  ANTHROPIC_API_KEY not found in environment variables")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        print("Get your key from: https://console.anthropic.com/")
        return

    try:
        # Create chatbot
        bot = RAGChatbot(api_key=api_key, provider="anthropic")

        # Load knowledge base
        docs = create_ml_knowledge_base()
        bot.load_knowledge_base(docs)

        # Test questions
        questions = [
            "What algorithm would you recommend for spam detection and why?",
            "How can I evaluate if my model will work on new data?",
            "What's the difference between precision and recall?"
        ]

        for question in questions:
            print(f"\n👤 User: {question}")
            response = bot.chat(question, use_rag=True)
            print(f"🤖 Bot: {response}")

        print("\n" + "="*70)

    except Exception as e:
        print(f"\n❌ Error: {e}")

def interactive_mode():
    """Interactive chatbot mode"""
    print("\n" + "="*70)
    print("💬 Interactive RAG Chatbot")
    print("="*70)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n⚠️  ANTHROPIC_API_KEY not set. Running vector search only mode.")
        print("\nType 'quit' to exit\n")

        store = SimpleVectorStore()
        docs = create_ml_knowledge_base()
        store.add_documents(docs)

        while True:
            query = input("You: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            results = store.search(query, top_k=2)
            print("\nRelevant information:")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['document']['text']}")
            print()

    else:
        bot = RAGChatbot(api_key=api_key)
        docs = create_ml_knowledge_base()
        bot.load_knowledge_base(docs)

        print("\nRAG Chatbot ready! Type 'quit' to exit\n")

        while True:
            query = input("You: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if not query:
                continue

            response = bot.chat(query, use_rag=True)
            print(f"\nBot: {response}\n")

# ============== MAIN ==============

if __name__ == "__main__":
    import sys

    print("🚀 RAG Chatbot Demo - Learn Generative AI Concepts")
    print("This demonstrates: Vector Search, Embeddings, LLM Integration, RAG")

    # Always run vector search demo (no API needed)
    if HAS_SKLEARN:
        demo_without_api()

    # Check if user wants full demo
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        if HAS_ANTHROPIC:
            demo_with_api()
        else:
            print("\n⚠️  Install anthropic to run full demo: pip install anthropic")
    elif len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        print("\n💡 Usage:")
        print("  python rag_chatbot.py              # Vector search demo only")
        print("  python rag_chatbot.py --full       # Full RAG with Claude API")
        print("  python rag_chatbot.py --interactive # Interactive chat mode")
