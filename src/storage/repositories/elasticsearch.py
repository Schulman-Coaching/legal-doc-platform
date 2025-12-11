"""
Elasticsearch Repository
========================
Async Elasticsearch repository for full-text and semantic search.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Any, Optional

from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk

from ..config import ElasticsearchConfig
from ..models import SearchDocument

logger = logging.getLogger(__name__)


class ElasticsearchRepository:
    """
    Elasticsearch repository for document search.

    Supports full-text search, semantic/vector search, and aggregations.
    """

    # Index mapping for legal documents
    INDEX_SETTINGS = {
        "settings": {
            "number_of_shards": 5,
            "number_of_replicas": 1,
            "refresh_interval": "1s",
            "analysis": {
                "analyzer": {
                    "legal_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",
                            "legal_synonyms",
                            "english_stemmer",
                            "english_possessive_stemmer"
                        ]
                    },
                    "citation_analyzer": {
                        "type": "custom",
                        "tokenizer": "keyword",
                        "filter": ["lowercase", "trim"]
                    },
                    "autocomplete_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "autocomplete_filter"]
                    }
                },
                "filter": {
                    "legal_synonyms": {
                        "type": "synonym",
                        "synonyms": [
                            "plaintiff,claimant,petitioner,complainant",
                            "defendant,respondent,appellee",
                            "court,tribunal,judiciary,bench",
                            "contract,agreement,covenant,pact",
                            "breach,violation,infringement",
                            "damages,compensation,indemnity",
                            "liability,responsibility,obligation",
                            "negligence,carelessness,fault",
                            "injunction,restraining order",
                            "deposition,testimony,statement"
                        ]
                    },
                    "english_stemmer": {
                        "type": "stemmer",
                        "language": "english"
                    },
                    "english_possessive_stemmer": {
                        "type": "stemmer",
                        "language": "possessive_english"
                    },
                    "autocomplete_filter": {
                        "type": "edge_ngram",
                        "min_gram": 2,
                        "max_gram": 20
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "title": {
                    "type": "text",
                    "analyzer": "legal_analyzer",
                    "fields": {
                        "keyword": {"type": "keyword"},
                        "autocomplete": {
                            "type": "text",
                            "analyzer": "autocomplete_analyzer",
                            "search_analyzer": "standard"
                        }
                    }
                },
                "content": {
                    "type": "text",
                    "analyzer": "legal_analyzer",
                    "term_vector": "with_positions_offsets"
                },
                "content_vector": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine"
                },
                "document_type": {"type": "keyword"},
                "practice_areas": {"type": "keyword"},
                "client_id": {"type": "keyword"},
                "matter_id": {"type": "keyword"},
                "classification": {"type": "keyword"},
                "tags": {"type": "keyword"},
                "persons": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword"}}
                },
                "organizations": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword"}}
                },
                "locations": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword"}}
                },
                "dates": {"type": "keyword"},
                "case_numbers": {
                    "type": "text",
                    "analyzer": "citation_analyzer",
                    "fields": {"keyword": {"type": "keyword"}}
                },
                "citations": {
                    "type": "text",
                    "analyzer": "citation_analyzer",
                    "fields": {"keyword": {"type": "keyword"}}
                },
                "monetary_amounts": {"type": "keyword"},
                "document_date": {"type": "date"},
                "created_at": {"type": "date"},
                "updated_at": {"type": "date"},
                "language": {"type": "keyword"},
                "suggest": {
                    "type": "completion",
                    "analyzer": "simple",
                    "preserve_separators": True,
                    "preserve_position_increments": True,
                    "max_input_length": 50
                }
            }
        }
    }

    def __init__(self, config: ElasticsearchConfig):
        self.config = config
        self._client: Optional[AsyncElasticsearch] = None
        self._documents_index = f"{config.index_prefix}_documents"
        self._entities_index = f"{config.index_prefix}_entities"

    async def connect(self) -> None:
        """Connect to Elasticsearch cluster."""
        try:
            auth = None
            if self.config.username and self.config.password:
                auth = (self.config.username, self.config.password)

            self._client = AsyncElasticsearch(
                hosts=self.config.hosts,
                basic_auth=auth,
                ca_certs=self.config.ca_cert_path,
                verify_certs=self.config.verify_certs,
                request_timeout=self.config.request_timeout,
            )

            # Test connection
            info = await self._client.info()
            logger.info(
                f"Connected to Elasticsearch {info['version']['number']} "
                f"at {self.config.hosts}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise

    async def disconnect(self) -> None:
        """Close Elasticsearch connection."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Disconnected from Elasticsearch")

    async def create_indices(self) -> None:
        """Create search indices if they don't exist."""
        if not self._client:
            raise RuntimeError("Elasticsearch client not connected")

        # Update settings with config values
        settings = self.INDEX_SETTINGS.copy()
        settings["settings"]["number_of_shards"] = self.config.number_of_shards
        settings["settings"]["number_of_replicas"] = self.config.number_of_replicas
        settings["settings"]["refresh_interval"] = self.config.refresh_interval

        # Create documents index
        if not await self._client.indices.exists(index=self._documents_index):
            await self._client.indices.create(
                index=self._documents_index,
                body=settings,
            )
            logger.info(f"Created index {self._documents_index}")
        else:
            logger.info(f"Index {self._documents_index} already exists")

    async def delete_indices(self) -> None:
        """Delete all indices (use with caution)."""
        if not self._client:
            raise RuntimeError("Elasticsearch client not connected")

        if await self._client.indices.exists(index=self._documents_index):
            await self._client.indices.delete(index=self._documents_index)
            logger.info(f"Deleted index {self._documents_index}")

    # Document Operations

    async def index_document(self, doc: SearchDocument) -> bool:
        """Index a document for search."""
        if not self._client:
            raise RuntimeError("Elasticsearch client not connected")

        try:
            # Build suggest field for autocomplete
            suggest_input = [doc.title] if doc.title else []
            if doc.persons:
                suggest_input.extend(doc.persons[:5])
            if doc.organizations:
                suggest_input.extend(doc.organizations[:5])

            body = doc.model_dump(exclude_none=True)
            if suggest_input:
                body["suggest"] = {"input": suggest_input}

            await self._client.index(
                index=self._documents_index,
                id=doc.id,
                document=body,
                refresh=True,
            )

            logger.debug(f"Indexed document {doc.id}")
            return True

        except Exception as e:
            logger.error(f"Failed to index document {doc.id}: {e}")
            return False

    async def bulk_index(self, docs: list[SearchDocument]) -> dict[str, int]:
        """Bulk index multiple documents."""
        if not self._client:
            raise RuntimeError("Elasticsearch client not connected")

        def generate_actions():
            for doc in docs:
                body = doc.model_dump(exclude_none=True)
                suggest_input = [doc.title] if doc.title else []
                if suggest_input:
                    body["suggest"] = {"input": suggest_input}

                yield {
                    "_index": self._documents_index,
                    "_id": doc.id,
                    "_source": body,
                }

        try:
            success, failed = await async_bulk(
                self._client,
                generate_actions(),
                raise_on_error=False,
            )
            logger.info(f"Bulk indexed {success} documents, {len(failed)} failed")
            return {"success": success, "failed": len(failed)}
        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")
            return {"success": 0, "failed": len(docs)}

    async def get_document(self, document_id: str) -> Optional[dict[str, Any]]:
        """Get a document by ID."""
        if not self._client:
            raise RuntimeError("Elasticsearch client not connected")

        try:
            result = await self._client.get(
                index=self._documents_index,
                id=document_id,
            )
            return result["_source"]
        except Exception:
            return None

    async def delete_document(self, document_id: str) -> bool:
        """Delete document from search index."""
        if not self._client:
            raise RuntimeError("Elasticsearch client not connected")

        try:
            await self._client.delete(
                index=self._documents_index,
                id=document_id,
                refresh=True,
            )
            logger.debug(f"Deleted document {document_id} from search index")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    async def update_document(
        self,
        document_id: str,
        updates: dict[str, Any],
    ) -> bool:
        """Partially update a document."""
        if not self._client:
            raise RuntimeError("Elasticsearch client not connected")

        try:
            await self._client.update(
                index=self._documents_index,
                id=document_id,
                doc=updates,
                refresh=True,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update document {document_id}: {e}")
            return False

    # Search Operations

    async def search(
        self,
        query: str,
        filters: Optional[dict[str, Any]] = None,
        from_: int = 0,
        size: int = 20,
        highlight: bool = True,
        sort: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """Full-text search with filters."""
        if not self._client:
            raise RuntimeError("Elasticsearch client not connected")

        start_time = datetime.utcnow()

        # Build query
        must = []
        filter_clauses = []

        if query and query.strip():
            must.append({
                "multi_match": {
                    "query": query,
                    "fields": [
                        "title^3",
                        "content",
                        "persons^2",
                        "organizations^2",
                        "case_numbers^2",
                        "citations"
                    ],
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                    "prefix_length": 2,
                }
            })

        # Add filters
        if filters:
            for field, value in filters.items():
                if value is None:
                    continue
                if isinstance(value, list):
                    filter_clauses.append({"terms": {field: value}})
                else:
                    filter_clauses.append({"term": {field: value}})

        es_query = {
            "bool": {
                "must": must if must else [{"match_all": {}}],
                "filter": filter_clauses,
            }
        }

        # Build request body
        body = {
            "query": es_query,
            "from": from_,
            "size": size,
        }

        # Add highlighting
        if highlight:
            body["highlight"] = {
                "fields": {
                    "content": {
                        "fragment_size": 200,
                        "number_of_fragments": 3,
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"],
                    },
                    "title": {}
                }
            }

        # Add sorting
        if sort:
            body["sort"] = sort
        else:
            body["sort"] = [
                {"_score": "desc"},
                {"created_at": "desc"}
            ]

        try:
            response = await self._client.search(
                index=self._documents_index,
                body=body,
            )

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            hits = []
            for hit in response["hits"]["hits"]:
                doc = hit["_source"]
                doc["_score"] = hit["_score"]
                if "highlight" in hit:
                    doc["_highlights"] = hit["highlight"]
                hits.append(doc)

            return {
                "total": response["hits"]["total"]["value"],
                "hits": hits,
                "took_ms": response["took"],
                "query_time_ms": duration_ms,
            }

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                "total": 0,
                "hits": [],
                "took_ms": 0,
                "error": str(e),
            }

    async def semantic_search(
        self,
        query_vector: list[float],
        filters: Optional[dict[str, Any]] = None,
        size: int = 20,
        min_score: float = 0.5,
    ) -> dict[str, Any]:
        """Semantic search using vector similarity."""
        if not self._client:
            raise RuntimeError("Elasticsearch client not connected")

        start_time = datetime.utcnow()

        # Build filter
        filter_clauses = []
        if filters:
            for field, value in filters.items():
                if value is None:
                    continue
                if isinstance(value, list):
                    filter_clauses.append({"terms": {field: value}})
                else:
                    filter_clauses.append({"term": {field: value}})

        try:
            response = await self._client.search(
                index=self._documents_index,
                knn={
                    "field": "content_vector",
                    "query_vector": query_vector,
                    "k": size,
                    "num_candidates": size * 10,
                    "filter": {"bool": {"filter": filter_clauses}} if filter_clauses else None,
                },
                size=size,
            )

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            hits = []
            for hit in response["hits"]["hits"]:
                if hit["_score"] >= min_score:
                    doc = hit["_source"]
                    doc["_score"] = hit["_score"]
                    hits.append(doc)

            return {
                "total": len(hits),
                "hits": hits,
                "took_ms": response["took"],
                "query_time_ms": duration_ms,
            }

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return {
                "total": 0,
                "hits": [],
                "took_ms": 0,
                "error": str(e),
            }

    async def hybrid_search(
        self,
        query: str,
        query_vector: list[float],
        filters: Optional[dict[str, Any]] = None,
        size: int = 20,
        text_weight: float = 0.5,
    ) -> dict[str, Any]:
        """Hybrid search combining text and semantic search."""
        if not self._client:
            raise RuntimeError("Elasticsearch client not connected")

        vector_weight = 1.0 - text_weight

        # Build filter
        filter_clauses = []
        if filters:
            for field, value in filters.items():
                if value is None:
                    continue
                if isinstance(value, list):
                    filter_clauses.append({"terms": {field: value}})
                else:
                    filter_clauses.append({"term": {field: value}})

        body = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^3", "content"],
                                "boost": text_weight,
                            }
                        }
                    ],
                    "filter": filter_clauses,
                }
            },
            "knn": {
                "field": "content_vector",
                "query_vector": query_vector,
                "k": size,
                "num_candidates": size * 10,
                "boost": vector_weight,
            },
            "size": size,
        }

        try:
            response = await self._client.search(
                index=self._documents_index,
                body=body,
            )

            hits = []
            for hit in response["hits"]["hits"]:
                doc = hit["_source"]
                doc["_score"] = hit["_score"]
                hits.append(doc)

            return {
                "total": response["hits"]["total"]["value"],
                "hits": hits,
                "took_ms": response["took"],
            }

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return {"total": 0, "hits": [], "error": str(e)}

    async def autocomplete(
        self,
        prefix: str,
        size: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Get autocomplete suggestions."""
        if not self._client:
            raise RuntimeError("Elasticsearch client not connected")

        try:
            # Completion suggester
            body = {
                "suggest": {
                    "doc-suggest": {
                        "prefix": prefix,
                        "completion": {
                            "field": "suggest",
                            "size": size,
                            "skip_duplicates": True,
                        }
                    }
                }
            }

            response = await self._client.search(
                index=self._documents_index,
                body=body,
            )

            suggestions = []
            for option in response["suggest"]["doc-suggest"][0]["options"]:
                suggestions.append({
                    "text": option["text"],
                    "score": option["_score"],
                    "id": option["_id"],
                })

            return suggestions

        except Exception as e:
            logger.error(f"Autocomplete failed: {e}")
            return []

    async def more_like_this(
        self,
        document_id: str,
        size: int = 10,
        min_term_freq: int = 1,
        min_doc_freq: int = 1,
    ) -> list[dict[str, Any]]:
        """Find similar documents."""
        if not self._client:
            raise RuntimeError("Elasticsearch client not connected")

        try:
            response = await self._client.search(
                index=self._documents_index,
                query={
                    "more_like_this": {
                        "fields": ["title", "content"],
                        "like": [{"_index": self._documents_index, "_id": document_id}],
                        "min_term_freq": min_term_freq,
                        "min_doc_freq": min_doc_freq,
                    }
                },
                size=size,
            )

            results = []
            for hit in response["hits"]["hits"]:
                doc = hit["_source"]
                doc["_score"] = hit["_score"]
                results.append(doc)

            return results

        except Exception as e:
            logger.error(f"More like this failed: {e}")
            return []

    # Aggregations

    async def aggregate(
        self,
        field: str,
        filters: Optional[dict[str, Any]] = None,
        size: int = 10,
    ) -> list[dict[str, Any]]:
        """Get aggregations (facets) for a field."""
        if not self._client:
            raise RuntimeError("Elasticsearch client not connected")

        filter_clauses = []
        if filters:
            for f, value in filters.items():
                if value is None:
                    continue
                if isinstance(value, list):
                    filter_clauses.append({"terms": {f: value}})
                else:
                    filter_clauses.append({"term": {f: value}})

        body = {
            "size": 0,
            "query": {
                "bool": {"filter": filter_clauses} if filter_clauses else {"match_all": {}}
            },
            "aggs": {
                "facet": {
                    "terms": {
                        "field": field,
                        "size": size,
                    }
                }
            }
        }

        try:
            response = await self._client.search(
                index=self._documents_index,
                body=body,
            )

            return [
                {"value": bucket["key"], "count": bucket["doc_count"]}
                for bucket in response["aggregations"]["facet"]["buckets"]
            ]

        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return []

    async def get_statistics(
        self,
        filters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Get search index statistics."""
        if not self._client:
            raise RuntimeError("Elasticsearch client not connected")

        filter_clauses = []
        if filters:
            for field, value in filters.items():
                if value is None:
                    continue
                if isinstance(value, list):
                    filter_clauses.append({"terms": {field: value}})
                else:
                    filter_clauses.append({"term": {field: value}})

        body = {
            "size": 0,
            "query": {
                "bool": {"filter": filter_clauses} if filter_clauses else {"match_all": {}}
            },
            "aggs": {
                "by_type": {"terms": {"field": "document_type", "size": 20}},
                "by_practice_area": {"terms": {"field": "practice_areas", "size": 20}},
                "by_classification": {"terms": {"field": "classification", "size": 10}},
                "date_histogram": {
                    "date_histogram": {
                        "field": "created_at",
                        "calendar_interval": "month",
                    }
                }
            }
        }

        try:
            response = await self._client.search(
                index=self._documents_index,
                body=body,
            )

            return {
                "total_documents": response["hits"]["total"]["value"],
                "by_type": {
                    b["key"]: b["doc_count"]
                    for b in response["aggregations"]["by_type"]["buckets"]
                },
                "by_practice_area": {
                    b["key"]: b["doc_count"]
                    for b in response["aggregations"]["by_practice_area"]["buckets"]
                },
                "by_classification": {
                    b["key"]: b["doc_count"]
                    for b in response["aggregations"]["by_classification"]["buckets"]
                },
                "monthly_counts": [
                    {"month": b["key_as_string"], "count": b["doc_count"]}
                    for b in response["aggregations"]["date_histogram"]["buckets"]
                ],
            }

        except Exception as e:
            logger.error(f"Statistics failed: {e}")
            return {}

    # Health Check

    async def health_check(self) -> dict[str, Any]:
        """Check Elasticsearch cluster health."""
        if not self._client:
            return {"status": "disconnected"}

        try:
            health = await self._client.cluster.health()
            stats = await self._client.indices.stats(index=self._documents_index)

            index_stats = stats["indices"].get(self._documents_index, {})
            primaries = index_stats.get("primaries", {})

            return {
                "status": health["status"],
                "cluster_name": health["cluster_name"],
                "number_of_nodes": health["number_of_nodes"],
                "active_shards": health["active_shards"],
                "index": {
                    "name": self._documents_index,
                    "docs_count": primaries.get("docs", {}).get("count", 0),
                    "size_bytes": primaries.get("store", {}).get("size_in_bytes", 0),
                }
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    # Cache key generation for search results
    @staticmethod
    def generate_cache_key(query: str, filters: Optional[dict], from_: int, size: int) -> str:
        """Generate a cache key for search results."""
        key_data = f"{query}:{json.dumps(filters or {}, sort_keys=True)}:{from_}:{size}"
        return hashlib.md5(key_data.encode()).hexdigest()


import json  # noqa: E402 (import at top would be better but keeping near usage)
