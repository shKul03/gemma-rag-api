"""
Hybrid search service combining semantic (ChromaDB) and keyword (Elasticsearch) search.
Merges results from both systems for improved retrieval accuracy.
"""

import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

from app.rag import rag_service
from app.elasticsearch_service import get_elasticsearch_service

logger = logging.getLogger(__name__)


class HybridSearchService:
    """
    Combines semantic search (ChromaDB) with keyword search (Elasticsearch).
    
    Features:
    - Parallel querying of both search engines
    - Score normalization and fusion
    - Deduplication with score aggregation
    - Configurable weighting between semantic and keyword
    """
    
    def __init__(
        self,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4
    ):
        """
        Initialize hybrid search service.
        
        Args:
            semantic_weight: Weight for semantic search scores (0-1)
            keyword_weight: Weight for keyword search scores (0-1)
            
        Note: Weights should sum to 1.0 for balanced scoring
        """
        self.rag_service = rag_service
        self.es_service = get_elasticsearch_service()
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        
        # Validate weights
        total_weight = semantic_weight + keyword_weight
        if not 0.99 <= total_weight <= 1.01:  # Allow small floating point error
            logger.warning(
                f"Weights sum to {total_weight:.2f}, not 1.0. "
                f"Using normalized weights."
            )
            self.semantic_weight = semantic_weight / total_weight
            self.keyword_weight = keyword_weight / total_weight
        
        logger.info(
            f"🔀 Hybrid search initialized: "
            f"semantic={self.semantic_weight:.2f}, "
            f"keyword={self.keyword_weight:.2f}"
        )
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        semantic_k: int = 20,
        keyword_k: int = 20,
        return_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword results.
        
        Args:
            query: Search query string
            top_k: Number of final results to return
            semantic_k: Number of results to fetch from ChromaDB
            keyword_k: Number of results to fetch from Elasticsearch
            return_metadata: Include metadata in results
            
        Returns:
            List of dicts with merged and ranked results
        """
        logger.info(f"🔍 Hybrid search: '{query[:50]}...'")
        
        # Step 1: Query both engines in parallel
        semantic_results = []
        keyword_results = []
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both searches
            semantic_future = executor.submit(
                self._search_semantic,
                query,
                semantic_k
            )
            keyword_future = executor.submit(
                self._search_keyword,
                query,
                keyword_k
            )
            
            # Collect results
            for future in as_completed([semantic_future, keyword_future]):
                try:
                    result = future.result()
                    if future == semantic_future:
                        semantic_results = result
                    else:
                        keyword_results = result
                except Exception as e:
                    logger.error(f"Search failed: {e}")
        
        logger.info(
            f"   Semantic: {len(semantic_results)} results, "
            f"Keyword: {len(keyword_results)} results"
        )
        
        # Step 2: Normalize scores
        semantic_normalized = self._normalize_scores(
            semantic_results,
            "semantic"
        )
        keyword_normalized = self._normalize_scores(
            keyword_results,
            "keyword"
        )
        
        # Step 3: Merge and deduplicate
        merged = self._merge_results(
            semantic_normalized,
            keyword_normalized
        )
        
        # Step 4: Sort by hybrid score and return top_k
        merged.sort(key=lambda x: x['hybrid_score'], reverse=True)
        top_results = merged[:top_k]
        
        logger.info(
            f"   ✅ Returned {len(top_results)} results "
            f"(from {len(merged)} unique)"
        )
        
        # Format results
        if not return_metadata:
            return [r['text'] for r in top_results]
        
        return top_results
    
    def _search_semantic(
        self,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Search using semantic similarity (ChromaDB).
        
        Returns:
            List of dicts with 'chunk_id', 'text', 'score', 'metadata'
        """
        try:
            chunks = self.rag_service.retrieve_context(
                query,
                top_k=top_k,
                return_metadata=True
            )
            
            results = []
            for chunk in chunks:
                result = {
                    'chunk_id': chunk.get('metadata', {}).get('chunk_id', 'unknown'),
                    'text': chunk['text'],
                    'score': chunk['relevance_score'],
                    'metadata': chunk['metadata'],
                    'source': 'semantic'
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def _search_keyword(
        self,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Search using keyword matching (Elasticsearch BM25).
        
        Returns:
            List of dicts with 'chunk_id', 'text', 'score', 'metadata'
        """
        try:
            results = self.es_service.search(query, top_k=top_k)
            
            # Standardize format
            formatted = []
            for result in results:
                formatted.append({
                    'chunk_id': result['chunk_id'],
                    'text': result['text'],
                    'score': result['score'],
                    'metadata': result['metadata'],
                    'source': 'keyword'
                })
            
            return formatted
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    def _normalize_scores(
        self,
        results: List[Dict[str, Any]],
        source: str
    ) -> List[Dict[str, Any]]:
        """
        Normalize scores to 0-1 range using min-max normalization.
        
        Args:
            results: List of search results with 'score' field
            source: 'semantic' or 'keyword' for logging
            
        Returns:
            Results with added 'normalized_score' field
        """
        if not results:
            return results
        
        scores = [r['score'] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        # Handle edge case where all scores are the same
        score_range = max_score - min_score
        if score_range == 0:
            for result in results:
                result['normalized_score'] = 1.0
        else:
            for result in results:
                normalized = (result['score'] - min_score) / score_range
                result['normalized_score'] = normalized
        
        return results
    
    def _merge_results(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge results from both sources, handling duplicates.
        
        Duplicates (same chunk_id) get combined scores:
        hybrid_score = semantic_weight * semantic_score + keyword_weight * keyword_score
        
        Args:
            semantic_results: Normalized semantic results
            keyword_results: Normalized keyword results
            
        Returns:
            Merged and deduplicated results with hybrid scores
        """
        merged_dict = {}
        
        # Add semantic results
        for result in semantic_results:
            chunk_id = result['chunk_id']
            merged_dict[chunk_id] = {
                'chunk_id': chunk_id,
                'text': result['text'],
                'metadata': result['metadata'],
                'semantic_score': result['normalized_score'],
                'keyword_score': 0.0,
                'sources': ['semantic']
            }
        
        # Add/merge keyword results
        for result in keyword_results:
            chunk_id = result['chunk_id']
            
            if chunk_id in merged_dict:
                # Duplicate: update keyword score and sources
                merged_dict[chunk_id]['keyword_score'] = result['normalized_score']
                merged_dict[chunk_id]['sources'].append('keyword')
            else:
                # New result from keyword search only
                merged_dict[chunk_id] = {
                    'chunk_id': chunk_id,
                    'text': result['text'],
                    'metadata': result['metadata'],
                    'semantic_score': 0.0,
                    'keyword_score': result['normalized_score'],
                    'sources': ['keyword']
                }
        
        # Calculate hybrid scores
        merged_list = []
        for chunk_id, data in merged_dict.items():
            hybrid_score = (
                self.semantic_weight * data['semantic_score'] +
                self.keyword_weight * data['keyword_score']
            )
            
            data['hybrid_score'] = hybrid_score
            merged_list.append(data)
        
        # Log statistics
        both_sources = sum(1 for d in merged_list if len(d['sources']) > 1)
        logger.info(
            f"   Merged: {len(merged_list)} unique chunks "
            f"({both_sources} found by both engines)"
        )
        
        return merged_list
    
    def compare_search_methods(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Compare results from semantic-only, keyword-only, and hybrid search.
        Useful for testing and evaluation.
        
        Args:
            query: Search query
            top_k: Number of results for each method
            
        Returns:
            Dict with results from all three methods and statistics
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"🔬 COMPARING SEARCH METHODS: '{query}'")
        logger.info(f"{'='*70}\n")
        
        # Semantic only
        semantic = self._search_semantic(query, top_k=top_k)
        logger.info(f"   Semantic-only: {len(semantic)} results")
        
        # Keyword only
        keyword = self._search_keyword(query, top_k=top_k)
        logger.info(f"   Keyword-only: {len(keyword)} results")
        
        # Hybrid
        hybrid = self.search(query, top_k=top_k, return_metadata=True)
        logger.info(f"   Hybrid: {len(hybrid)} results\n")
        
        # Calculate overlap
        semantic_ids = {r['chunk_id'] for r in semantic}
        keyword_ids = {r['chunk_id'] for r in keyword}
        hybrid_ids = {r['chunk_id'] for r in hybrid}
        
        overlap = semantic_ids & keyword_ids
        
        return {
            'query': query,
            'semantic_only': semantic,
            'keyword_only': keyword,
            'hybrid': hybrid,
            'statistics': {
                'semantic_count': len(semantic),
                'keyword_count': len(keyword),
                'hybrid_count': len(hybrid),
                'semantic_keyword_overlap': len(overlap),
                'unique_to_semantic': len(semantic_ids - keyword_ids),
                'unique_to_keyword': len(keyword_ids - semantic_ids)
            }
        }


# Singleton instance
_hybrid_search_service: Optional[HybridSearchService] = None


def get_hybrid_search_service(
    semantic_weight: float = 0.6,
    keyword_weight: float = 0.4
) -> HybridSearchService:
    """
    Get or create singleton hybrid search service instance.
    
    Args:
        semantic_weight: Weight for semantic search (0-1)
        keyword_weight: Weight for keyword search (0-1)
        
    Returns:
        HybridSearchService instance
    """
    global _hybrid_search_service
    if _hybrid_search_service is None:
        _hybrid_search_service = HybridSearchService(
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight
        )
    return _hybrid_search_service
