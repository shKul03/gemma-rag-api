"""
Source Attribution Service
Handles source extraction, confidence scoring, and citation formatting for RAG responses
"""
from typing import List, Dict, Optional
from datetime import datetime


class SourceAttribution:
    def __init__(self, supabase_url: Optional[str] = None):
        """
        Initialize attribution service
        
        Args:
            supabase_url: Optional Supabase URL for generating document links
        """
        self.supabase_url = supabase_url
    
    def extract_sources(self, chunks: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Extract unique source metadata from retrieved chunks
        
        Args:
            chunks: List of {text, metadata, distance/score}
        
        Returns:
            List of formatted source citations
        """
        sources = []
        seen_sources = set()
        
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            source_file = metadata.get('source', 'Unknown')
            
            # Avoid duplicate sources (same document)
            if source_file in seen_sources:
                continue
            seen_sources.add(source_file)
            
            # Build source information
            source_info = {
                'document': source_file,
                'last_updated': metadata.get('last_updated', 'N/A'),
                'file_type': metadata.get('file_type', 'unknown'),
                'relevance_score': round(chunk.get('relevance_score', 0.0), 3),
                'chunk_index': metadata.get('chunk_index', 'N/A'),
                'total_chunks': metadata.get('total_chunks', 'N/A')
            }
            
            # Add document URL if available
            if self.supabase_url:
                source_info['download_url'] = self.generate_document_url(source_file)
            
            sources.append(source_info)
        
        return sources
    
    def calculate_confidence(
        self, 
        chunks: List[Dict[str, any]],
        answer: str
    ) -> Dict[str, any]:
        """
        Calculate confidence score for the generated answer
        
        Based on:
        - Retrieval relevance scores (lower distance = higher relevance)
        - Number of supporting chunks
        - Score distribution (consistency across chunks)
        
        Args:
            chunks: Retrieved chunks with similarity scores
            answer: Generated answer text
        
        Returns:
            Dict with score, level, and supporting metrics
        """
        if not chunks:
            return {
                'score': 0.0,
                'level': 'none',
                'message': 'No relevant context found'
            }
        
        # Extract relevance scores (convert distance to similarity: 1 - distance)
        # ChromaDB returns 'distance' - lower is better
        scores = [c.get('relevance_score', 0.0) for c in chunks]
        
        if not scores:
            return {
                'score': 0.5,
                'level': 'unknown',
                'message': 'Unable to calculate confidence'
            }
        
        avg_score = sum(scores) / len(scores)
        top_score = max(scores)
        score_variance = max(scores) - min(scores) if len(scores) > 1 else 0
        
        # Confidence calculation logic
        # High variance means inconsistent matches
        if top_score > 0.85 and len(chunks) >= 3 and score_variance < 0.3:
            confidence_level = 'high'
            confidence_score = 0.9
            message = '✅ High confidence - answer well-supported by documents'
        elif top_score > 0.70 and len(chunks) >= 2:
            confidence_level = 'medium'
            confidence_score = 0.7
            message = '⚠️ Medium confidence - answer reasonably supported'
        elif top_score > 0.50:
            confidence_level = 'low'
            confidence_score = 0.5
            message = '⚠️ Low confidence - limited supporting evidence'
        else:
            confidence_level = 'very_low'
            confidence_score = 0.3
            message = '❌ Very low confidence - please verify with HR'
        
        return {
            'score': round(confidence_score, 2),
            'level': confidence_level,
            'message': message,
            'top_relevance': round(top_score, 3),
            'avg_relevance': round(avg_score, 3),
            'num_sources': len(chunks),
            'score_variance': round(score_variance, 3)
        }
    
    def generate_document_url(self, document_name: str) -> Optional[str]:
        """
        Generate URL for document access
        
        Args:
            document_name: Name of the source document
        
        Returns:
            URL string or None
        """
        if not self.supabase_url:
            return None
        
        # Generate public URL (in production, use signed URLs with expiration)
        doc_path = f"hr_policies/{document_name}"
        return f"{self.supabase_url}/storage/v1/object/public/rag_docs/{doc_path}"
    
    def format_citation(
        self, 
        answer: str,
        chunks: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """
        Format complete answer with citations, sources, and confidence
        
        Args:
            answer: Generated answer text
            chunks: Retrieved chunks with metadata
        
        Returns:
            Complete response with answer, confidence, and sources
        """
        sources = self.extract_sources(chunks)
        confidence = self.calculate_confidence(chunks, answer)
        
        return {
            'answer': answer,
            'confidence': confidence,
            'sources': sources,
            'num_sources': len(sources),
            'timestamp': datetime.utcnow().isoformat()
        }


# Global instance
attribution_service = SourceAttribution(
    supabase_url="https://edsvrikgfibjibcuhcpz.supabase.co"
)
