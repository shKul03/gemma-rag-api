"""
Elasticsearch service for keyword search using BM25 algorithm.
Complements ChromaDB's semantic search for hybrid retrieval.
"""

from elasticsearch import Elasticsearch
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ElasticsearchService:
    """
    Service for managing Elasticsearch indexing and keyword search.
    
    Features:
    - Document indexing with metadata
    - BM25-based keyword search
    - Configurable index settings
    - Health checking
    """
    
    def __init__(
        self,
        host: str = "http://localhost:9200",
        index_name: str = "hr_documents"
    ):
        """
        Initialize Elasticsearch client.
        
        Args:
            host: Elasticsearch host URL
            index_name: Name of the index to use
        """
        self.es = Elasticsearch([host])
        self.index_name = index_name
        
        # Verify connection
        if self.es.ping():
            logger.info(f"✅ Connected to Elasticsearch at {host}")
        else:
            logger.error(f"❌ Failed to connect to Elasticsearch at {host}")
            raise ConnectionError(f"Cannot connect to Elasticsearch at {host}")
    
    def create_index(self, force_recreate: bool = False) -> bool:
        """
        Create index with optimized settings for HR document search.
        
        Settings:
        - BM25 similarity for keyword matching
        - Text fields analyzed with standard analyzer
        - Metadata fields as keywords for filtering
        
        Args:
            force_recreate: If True, delete existing index and recreate
            
        Returns:
            True if index created/exists, False otherwise
        """
        try:
            # Check if index exists
            if self.es.indices.exists(index=self.index_name):
                if force_recreate:
                    logger.info(f"Deleting existing index: {self.index_name}")
                    self.es.indices.delete(index=self.index_name)
                else:
                    logger.info(f"Index '{self.index_name}' already exists")
                    return True
            
            # Index settings and mappings
            index_config = {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {
                        "analyzer": {
                            "default": {
                                "type": "standard"
                            }
                        }
                    },
                    "similarity": {
                        "default": {
                            "type": "BM25",
                            "k1": 1.2,
                            "b": 0.75
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "text": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "chunk_id": {
                            "type": "keyword"
                        },
                        "source": {
                            "type": "keyword"
                        },
                        "chunk_index": {
                            "type": "integer"
                        },
                        "total_chunks": {
                            "type": "integer"
                        },
                        "last_updated": {
                            "type": "date",
                            "format": "strict_date_optional_time||epoch_millis"
                        },
                        "file_size": {
                            "type": "integer"
                        },
                        "file_type": {
                            "type": "keyword"
                        },
                        "ingestion_timestamp": {
                            "type": "date",
                            "format": "strict_date_optional_time||epoch_millis"
                        }
                    }
                }
            }
            
            # Create index
            self.es.indices.create(index=self.index_name, body=index_config)
            logger.info(f"✅ Created index: {self.index_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating index: {e}")
            return False
    
    def index_document(
        self,
        chunk_id: str,
        text: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Index a single document chunk.
        
        Args:
            chunk_id: Unique identifier for the chunk (matches ChromaDB ID)
            text: The text content to index
            metadata: Additional metadata (source, chunk_index, etc.)
            
        Returns:
            True if indexed successfully, False otherwise
        """
        try:
            doc = {
                "text": text,
                "chunk_id": chunk_id,
                **metadata
            }
            
            self.es.index(
                index=self.index_name,
                id=chunk_id,
                document=doc
            )
            return True
            
        except Exception as e:
            logger.error(f"❌ Error indexing document {chunk_id}: {e}")
            return False
    
    def bulk_index_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Bulk index multiple documents.
        
        Args:
            documents: List of dicts with 'chunk_id', 'text', and 'metadata' keys
            
        Returns:
            Dict with 'success' and 'failed' counts
        """
        from elasticsearch.helpers import bulk
        
        success_count = 0
        failed_count = 0
        
        try:
            # Prepare bulk actions
            actions = []
            for doc in documents:
                action = {
                    "_index": self.index_name,
                    "_id": doc["chunk_id"],
                    "_source": {
                        "text": doc["text"],
                        "chunk_id": doc["chunk_id"],
                        **doc.get("metadata", {})
                    }
                }
                actions.append(action)
            
            # Execute bulk indexing
            success, failed = bulk(self.es, actions, raise_on_error=False)
            success_count = success
            failed_count = len(failed) if isinstance(failed, list) else 0
            
            logger.info(f"✅ Bulk indexed {success_count} documents ({failed_count} failed)")
            
        except Exception as e:
            logger.error(f"❌ Error in bulk indexing: {e}")
        
        return {
            "success": success_count,
            "failed": failed_count
        }
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform BM25 keyword search.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Optional metadata filters (e.g., {'source': 'handbook.pdf'})
            
        Returns:
            List of dicts with 'chunk_id', 'text', 'metadata', and 'score'
        """
        try:
            # Build query
            must_clauses = [
                {
                    "match": {
                        "text": {
                            "query": query,
                            "operator": "or"
                        }
                    }
                }
            ]
            
            # Add filters if provided
            filter_clauses = []
            if filters:
                for field, value in filters.items():
                    filter_clauses.append({
                        "term": {field: value}
                    })
            
            search_query = {
                "query": {
                    "bool": {
                        "must": must_clauses,
                        "filter": filter_clauses
                    }
                },
                "size": top_k
            }
            
            # Execute search
            response = self.es.search(index=self.index_name, body=search_query)
            
            # Format results
            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                result = {
                    "chunk_id": hit["_id"],
                    "text": source["text"],
                    "score": hit["_score"],
                    "metadata": {
                        k: v for k, v in source.items()
                        if k not in ["text", "chunk_id"]
                    }
                }
                results.append(result)
            
            logger.info(f"🔍 Found {len(results)} results for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error searching: {e}")
            return []
    
    def get_document_count(self) -> int:
        """
        Get total number of documents in the index.
        
        Returns:
            Document count
        """
        try:
            response = self.es.count(index=self.index_name)
            count = response["count"]
            logger.info(f"📊 Index contains {count} documents")
            return count
        except Exception as e:
            logger.error(f"❌ Error getting document count: {e}")
            return 0
    
    def delete_index(self) -> bool:
        """
        Delete the index.
        
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            if self.es.indices.exists(index=self.index_name):
                self.es.indices.delete(index=self.index_name)
                logger.info(f"🗑️ Deleted index: {self.index_name}")
                return True
            else:
                logger.warning(f"Index '{self.index_name}' does not exist")
                return False
        except Exception as e:
            logger.error(f"❌ Error deleting index: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check Elasticsearch health and index status.
        
        Returns:
            Dict with health information
        """
        try:
            # Cluster health
            cluster_health = self.es.cluster.health()
            
            # Index stats
            index_exists = self.es.indices.exists(index=self.index_name)
            doc_count = 0
            
            if index_exists:
                stats = self.es.indices.stats(index=self.index_name)
                doc_count = stats["indices"][self.index_name]["total"]["docs"]["count"]
            
            return {
                "cluster_status": cluster_health["status"],
                "cluster_name": cluster_health["cluster_name"],
                "index_exists": index_exists,
                "document_count": doc_count,
                "elasticsearch_version": self.es.info()["version"]["number"]
            }
            
        except Exception as e:
            logger.error(f"❌ Error checking health: {e}")
            return {
                "error": str(e),
                "cluster_status": "unavailable"
            }


# Singleton instance
_es_service: Optional[ElasticsearchService] = None


def get_elasticsearch_service(
    host: str = "http://localhost:9200",
    index_name: str = "hr_documents"
) -> ElasticsearchService:
    """
    Get or create singleton Elasticsearch service instance.
    
    Args:
        host: Elasticsearch host URL
        index_name: Index name to use
        
    Returns:
        ElasticsearchService instance
    """
    global _es_service
    if _es_service is None:
        _es_service = ElasticsearchService(host=host, index_name=index_name)
    return _es_service
