import asyncio
import logging
import time
import uuid
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import anthropic

from app.config import settings
from app.models.schemas import (
    QueryResponse, 
    Citation, 
    SearchResult, 
    ProcessedDocument,
    SourceType,
    CitationType
)

logger = logging.getLogger(__name__)

class RAGService:
    """Core RAG service for intelligent information retrieval and generation"""
    
    def __init__(self):
        self.genai_client = None
        self.claude_client = None
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.citation_cache = {}
        
    async def initialize(self):
        """Initialize the RAG service components"""
        try:
            # Initialize AI clients
            await self._initialize_ai_clients()
            
            # Initialize embedding model
            await self._initialize_embedding_model()
            
            # Initialize vector database
            await self._initialize_vector_db()
            
            logger.info("RAG service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            raise
    
    async def _initialize_ai_clients(self):
        """Initialize AI clients based on configuration"""
        try:
            if settings.ai_provider == "gemini" and settings.gemini_api_key:
                genai.configure(api_key=settings.gemini_api_key)
                self.genai_client = genai.GenerativeModel(settings.gemini_chat_model)
                logger.info("Gemini client initialized for RAG")
            elif settings.ai_provider == "claude" and settings.claude_api_key:
                self.claude_client = anthropic.AsyncAnthropic(api_key=settings.claude_api_key)
                logger.info("Claude client initialized for RAG")
            else:
                logger.warning("No AI client available for text generation")
        except Exception as e:
            logger.error(f"Failed to initialize AI clients: {e}")
            raise
    
    async def _initialize_embedding_model(self):
        """Initialize the sentence transformer model for embeddings"""
        try:
            loop = asyncio.get_event_loop()
            self.embedding_model = await loop.run_in_executor(
                None, 
                lambda: SentenceTransformer(settings.embedding_model)
            )
            logger.info(f"Embedding model loaded: {settings.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    async def _initialize_vector_db(self):
        """Initialize ChromaDB vector database"""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=settings.chroma_persist_directory
            )
            
            self.collection = self.chroma_client.get_or_create_collection(
                name="agentic_rag_documents"
            )
            
            logger.info("ChromaDB initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            raise
    
    async def index_document(self, document: ProcessedDocument):
        """Index a processed document in the vector database"""
        if not self.collection or not self.embedding_model:
            raise Exception("RAG service not properly initialized")
        
        try:
            start_time = time.time()
            
            # Prepare chunks for indexing
            texts = []
            metadatas = []
            ids = []
            
            for chunk in document.text_chunks:
                texts.append(chunk.content)
                metadatas.append({
                    "document_id": document.document_id,
                    "filename": document.filename,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "content_type": chunk.metadata.get("content_type", "text"),
                    "indexed_at": datetime.now().isoformat()
                })
                ids.append(chunk.id)
            
            # Generate embeddings
            embeddings = await self._generate_embeddings(texts)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            processing_time = int((time.time() - start_time) * 1000)
            logger.info(f"Document indexed: {document.filename} in {processing_time}ms")
            
        except Exception as e:
            logger.error(f"Document indexing failed: {e}")
            raise
    
    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Search the local document collection"""
        if not self.collection or not self.embedding_model:
            logger.warning("RAG service not properly initialized")
            return []
        
        try:
            logger.info(f"Starting RAG search for query: '{query}' with num_results: {num_results}")
            
            # Check collection status
            collection_count = self.collection.count()
            logger.info(f"Collection contains {collection_count} documents")
            
            if collection_count == 0:
                logger.warning("No documents in collection for RAG search")
                return []
            
            # Generate query embedding
            query_embedding = await self._generate_embeddings([query])
            logger.info(f"Generated query embedding of length {len(query_embedding[0])}")
            
            # Search collection
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=num_results,
                include=["documents", "metadatas", "distances"]
            )
            
            logger.info(f"ChromaDB returned {len(results['documents'][0]) if results['documents'] else 0} results")
            
            # Convert to SearchResult objects
            search_results = []
            
            if not results["documents"] or not results["documents"][0]:
                logger.warning("No search results returned from ChromaDB")
                return []
            
            for i in range(len(results["documents"][0])):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                
                # Fix confidence calculation for cosine distance
                # ChromaDB cosine distance ranges from 0 to 2, normalize to 0-1 range
                confidence_score = max(0.0, 1.0 - (distance / 2.0))
                
                logger.info(f"Result {i+1}: distance={distance:.4f}, confidence={confidence_score:.4f}, filename={metadata.get('filename', 'Unknown')}")
                
                search_result = SearchResult(
                    id=results["ids"][0][i],
                    title=metadata.get("filename", "Unknown Document"),
                    content=results["documents"][0][i],
                    source_type=SourceType.DOCUMENT,
                    url=None,
                    confidence_score=confidence_score,
                    metadata=metadata
                )
                search_results.append(search_result)
            
            logger.info(f"RAG search completed: {len(search_results)} results found")
            return search_results
            
        except Exception as e:
            logger.error(f"Local search failed: {e}")
            return []
    
    async def generate_response(self, query: str, rag_results: List[SearchResult],
                              web_results: List[SearchResult], 
                              drive_results: List[SearchResult]) -> QueryResponse:
        """Generate a comprehensive response using all available sources"""
        start_time = time.time()
        
        try:
            # Combine all results
            all_results = rag_results + web_results + drive_results
            all_results.sort(key=lambda x: x.confidence_score, reverse=True)
            all_results = all_results[:10]  # Limit to top 10
            
            # Generate citations
            citations = await self._create_citations(all_results, query)
            
            # Generate answer using LLM
            answer = await self._generate_answer(query, all_results)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return QueryResponse(
                answer=answer,
                citations=citations,
                search_results=all_results,
                confidence_score=0.8,  # Default confidence
                processing_time_ms=processing_time,
                sources_used={
                    SourceType.DOCUMENT: len(rag_results),
                    SourceType.WEB: len(web_results),
                    SourceType.GOOGLE_DRIVE: len(drive_results)
                },
                metadata={"query": query, "timestamp": datetime.now().isoformat()}
            )
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return QueryResponse(
                answer="I encountered an error while processing your query.",
                citations=[],
                search_results=[],
                confidence_score=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000),
                sources_used={},
                metadata={"error": str(e)}
            )
    
    async def _create_citations(self, results: List[SearchResult], 
                              query: str) -> List[Citation]:
        """Create citations from search results"""
        citations = []
        
        for i, result in enumerate(results[:5]):  # Top 5 citations
            citation_id = f"cite_{i+1}_{result.id}"
            
            citation = Citation(
                id=citation_id,
                source_type=result.source_type,
                citation_type=CitationType.TEXT,
                title=result.title,
                content=result.content[:200] + "..." if len(result.content) > 200 else result.content,
                url=result.url,
                page_number=result.metadata.get("page_number"),
                confidence_score=result.confidence_score,
                metadata=result.metadata
            )
            
            citations.append(citation)
            self.citation_cache[citation_id] = result
        
        return citations
    
    async def _generate_answer(self, query: str, results: List[SearchResult]) -> str:
        """Generate an answer using the configured language model"""
        if not self.genai_client and not self.claude_client:
            return "I don't have access to a language model to generate responses."
        
        try:
            # Prepare context
            context_parts = []
            for i, result in enumerate(results[:3]):
                context_parts.append(f"[{i+1}] {result.content}")
            
            context = "\n\n".join(context_parts)
            
            prompt = f"""You are a helpful AI assistant. Based on the provided sources, answer the user's question comprehensively and accurately.

User Question: {query}

Available Sources:
{context}

Instructions:
- Provide a detailed, helpful answer based on the sources
- Use citations [1], [2], [3] to reference specific sources
- If sources don't fully answer the question, acknowledge what's covered and what's missing
- Be specific and informative rather than giving generic responses
- Don't just say "Projects" - explain what the sources actually contain
- If the sources contain technical information, explain it clearly

Answer:"""
            
            if settings.ai_provider == "gemini" and self.genai_client:
                return await self._generate_with_gemini(prompt)
            elif settings.ai_provider == "claude" and self.claude_client:
                return await self._generate_with_claude(prompt)
            else:
                return "No AI provider available for text generation."
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "I encountered an error while generating the response."
    
    async def _generate_with_gemini(self, prompt: str) -> str:
        """Generate response using Gemini"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.genai_client.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=500
                    )
                )
            )
            
            return response.text if response.text else "Unable to generate response"
            
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return "Error generating response with Gemini"
    
    async def _generate_with_claude(self, prompt: str) -> str:
        """Generate response using Claude"""
        try:
            message = await self.claude_client.messages.create(
                model=settings.claude_chat_model,
                max_tokens=500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return message.content[0].text if message.content else "Unable to generate response"
            
        except Exception as e:
            logger.error(f"Claude generation failed: {e}")
            return "Error generating response with Claude"
    
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        try:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.embedding_model.encode(texts, convert_to_tensor=False)
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    async def get_citation_content(self, citation_id: str) -> Dict[str, Any]:
        """Get detailed content for a specific citation"""
        if citation_id not in self.citation_cache:
            raise Exception("Citation not found")
        
        result = self.citation_cache[citation_id]
        return {
            "title": result.title,
            "content": result.content,
            "source_type": result.source_type.value,
            "url": result.url,
            "metadata": result.metadata
        }
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all indexed documents"""
        if not self.collection:
            return []
        
        try:
            results = self.collection.get(include=["metadatas"])
            documents = {}
            
            for metadata in results["metadatas"]:
                doc_id = metadata.get("document_id")
                if doc_id not in documents:
                    documents[doc_id] = {
                        "document_id": doc_id,
                        "filename": metadata.get("filename"),
                        "indexed_at": metadata.get("indexed_at"),
                        "chunks": 0
                    }
                documents[doc_id]["chunks"] += 1
            
            return list(documents.values())
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []
    
    async def delete_document(self, document_id: str):
        """Delete a document from the vector database"""
        if not self.collection:
            raise Exception("Vector database not available")
        
        try:
            results = self.collection.get(
                where={"document_id": document_id},
                include=["ids"]
            )
            
            chunk_ids = results["ids"]
            if chunk_ids:
                self.collection.delete(ids=chunk_ids)
                logger.info(f"Deleted document {document_id}")
                
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            raise
    
    async def is_healthy(self) -> bool:
        """Check if the RAG service is healthy"""
        try:
            return (
                self.embedding_model is not None and
                self.collection is not None and
                (self.genai_client is not None or self.claude_client is not None)
            )
        except Exception:
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        self.citation_cache.clear() 