"""
Knowledge Graph Service
========================
Build and query knowledge graphs using Neo4j.
"""

import hashlib
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


class KnowledgeGraphService:
    """
    Service for building and querying knowledge graphs.
    Stores documents, entities, and their relationships in Neo4j.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        Initialize knowledge graph service.

        Args:
            uri: Neo4j URI (default: from NEO4J_URI env var)
            user: Neo4j username (default: from NEO4J_USER env var)
            password: Neo4j password (default: from NEO4J_PASSWORD env var)
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")

        self._driver = None
        self._connected = False

    async def connect(self) -> bool:
        """Establish connection to Neo4j."""
        try:
            from neo4j import AsyncGraphDatabase

            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            # Verify connection
            async with self._driver.session() as session:
                await session.run("RETURN 1")

            self._connected = True
            logger.info(f"Connected to Neo4j at {self.uri}")
            return True

        except ImportError:
            logger.warning("neo4j package not installed, using mock mode")
            self._connected = False
            return False

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self._connected = False
            return False

    async def close(self) -> None:
        """Close Neo4j connection."""
        if self._driver:
            await self._driver.close()
            self._connected = False

    async def _run_query(
        self,
        query: str,
        parameters: Optional[dict] = None,
    ) -> list[dict[str, Any]]:
        """Run a Cypher query."""
        if not self._connected:
            await self.connect()

        if not self._connected or not self._driver:
            logger.warning("Neo4j not connected, returning empty result")
            return []

        async with self._driver.session() as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records

    def _generate_entity_id(self, entity_type: str, value: str) -> str:
        """Generate deterministic ID for an entity."""
        return hashlib.sha256(f"{entity_type}:{value}".encode()).hexdigest()[:16]

    async def add_document(
        self,
        document_id: str,
        title: Optional[str] = None,
        document_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Add a document node to the graph.

        Args:
            document_id: Unique document identifier
            title: Document title
            document_type: Type of document (contract, memo, etc.)
            metadata: Additional metadata

        Returns:
            Created node information
        """
        metadata = metadata or {}

        query = """
        MERGE (d:Document {id: $id})
        SET d.title = $title,
            d.document_type = $document_type,
            d.created_at = datetime(),
            d += $metadata
        RETURN d
        """

        result = await self._run_query(query, {
            "id": document_id,
            "title": title,
            "document_type": document_type,
            "metadata": metadata,
        })

        return {"document_id": document_id, "created": True}

    async def add_entity(
        self,
        entity_type: str,
        value: str,
        normalized_value: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Add an entity node to the graph.

        Args:
            entity_type: Type of entity (Person, Organization, etc.)
            value: Entity value/name
            normalized_value: Normalized form of the value
            metadata: Additional metadata

        Returns:
            Entity ID
        """
        entity_id = self._generate_entity_id(entity_type, value)
        metadata = metadata or {}

        # Use entity_type as label
        label = entity_type.replace(" ", "_").title()

        query = f"""
        MERGE (e:{label} {{id: $id}})
        SET e.value = $value,
            e.normalized_value = $normalized_value,
            e.entity_type = $entity_type,
            e += $metadata
        RETURN e
        """

        await self._run_query(query, {
            "id": entity_id,
            "value": value,
            "normalized_value": normalized_value or value,
            "entity_type": entity_type,
            "metadata": metadata,
        })

        return entity_id

    async def link_document_entity(
        self,
        document_id: str,
        entity_id: str,
        relationship_type: str = "CONTAINS",
        properties: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Create relationship between document and entity.

        Args:
            document_id: Document node ID
            entity_id: Entity node ID
            relationship_type: Type of relationship
            properties: Relationship properties

        Returns:
            Success status
        """
        properties = properties or {}

        query = f"""
        MATCH (d:Document {{id: $doc_id}})
        MATCH (e {{id: $entity_id}})
        MERGE (d)-[r:{relationship_type}]->(e)
        SET r += $props
        RETURN r
        """

        await self._run_query(query, {
            "doc_id": document_id,
            "entity_id": entity_id,
            "props": properties,
        })

        return True

    async def add_document_to_graph(
        self,
        document_id: str,
        entities: list[dict[str, Any]],
        clauses: list[dict[str, Any]],
        citations: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> dict[str, int]:
        """
        Add document and all its extracted data to knowledge graph.

        Args:
            document_id: Document identifier
            entities: Extracted entities
            clauses: Detected clauses
            citations: Legal citations
            metadata: Document metadata

        Returns:
            Count of nodes and relationships created
        """
        nodes_created = 0
        relationships_created = 0

        # Create document node
        await self.add_document(
            document_id=document_id,
            title=metadata.get("title"),
            document_type=metadata.get("document_type"),
            metadata=metadata,
        )
        nodes_created += 1

        # Add entities
        for entity in entities:
            entity_id = await self.add_entity(
                entity_type=entity.get("type", "Entity"),
                value=entity.get("value", ""),
                normalized_value=entity.get("normalized_value"),
                metadata={
                    "confidence": entity.get("confidence", 0.0),
                    "position": entity.get("position"),
                },
            )
            nodes_created += 1

            await self.link_document_entity(
                document_id=document_id,
                entity_id=entity_id,
                relationship_type="CONTAINS_ENTITY",
                properties={"confidence": entity.get("confidence", 0.0)},
            )
            relationships_created += 1

        # Add clauses
        for clause in clauses:
            clause_id = self._generate_entity_id("Clause", f"{document_id}:{clause.get('type')}")

            query = """
            MERGE (c:Clause {id: $id})
            SET c.clause_type = $clause_type,
                c.risk_level = $risk_level,
                c.content = $content
            WITH c
            MATCH (d:Document {id: $doc_id})
            MERGE (d)-[:HAS_CLAUSE]->(c)
            RETURN c
            """

            await self._run_query(query, {
                "id": clause_id,
                "clause_type": clause.get("type"),
                "risk_level": clause.get("risk_level"),
                "content": clause.get("content", "")[:500],
                "doc_id": document_id,
            })
            nodes_created += 1
            relationships_created += 1

        # Add citations
        for citation in citations:
            citation_id = self._generate_entity_id("Citation", citation.get("normalized", citation.get("raw_text", "")))

            query = """
            MERGE (c:Citation {id: $id})
            SET c.citation_type = $citation_type,
                c.normalized = $normalized,
                c.raw_text = $raw_text
            WITH c
            MATCH (d:Document {id: $doc_id})
            MERGE (d)-[:CITES]->(c)
            RETURN c
            """

            await self._run_query(query, {
                "id": citation_id,
                "citation_type": citation.get("type"),
                "normalized": citation.get("normalized"),
                "raw_text": citation.get("raw_text"),
                "doc_id": document_id,
            })
            nodes_created += 1
            relationships_created += 1

        logger.info(f"Added document {document_id} to graph: {nodes_created} nodes, {relationships_created} relationships")

        return {
            "nodes_created": nodes_created,
            "relationships_created": relationships_created,
        }

    async def find_related_documents(
        self,
        document_id: str,
        relationship_types: Optional[list[str]] = None,
        max_depth: int = 2,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Find documents related to a given document through shared entities.

        Args:
            document_id: Source document ID
            relationship_types: Types of relationships to traverse
            max_depth: Maximum path length
            limit: Maximum results

        Returns:
            List of related documents with relationship info
        """
        query = """
        MATCH (d:Document {id: $doc_id})-[*1..%d]-(related:Document)
        WHERE related.id <> $doc_id
        WITH related, count(*) as connection_count
        ORDER BY connection_count DESC
        LIMIT $limit
        RETURN related.id as document_id,
               related.title as title,
               related.document_type as document_type,
               connection_count
        """ % max_depth

        results = await self._run_query(query, {
            "doc_id": document_id,
            "limit": limit,
        })

        return results

    async def find_entity_network(
        self,
        entity_value: str,
        entity_type: Optional[str] = None,
        max_depth: int = 2,
    ) -> dict[str, Any]:
        """
        Find all documents and related entities for an entity.

        Args:
            entity_value: Entity value to search for
            entity_type: Optional entity type filter
            max_depth: Maximum relationship depth

        Returns:
            Network of related documents and entities
        """
        entity_id = self._generate_entity_id(entity_type or "Entity", entity_value)

        # Find documents containing this entity
        doc_query = """
        MATCH (e {id: $entity_id})<-[:CONTAINS_ENTITY]-(d:Document)
        RETURN d.id as document_id, d.title as title
        """

        documents = await self._run_query(doc_query, {"entity_id": entity_id})

        # Find related entities
        related_query = """
        MATCH (e {id: $entity_id})<-[:CONTAINS_ENTITY]-(d:Document)-[:CONTAINS_ENTITY]->(related)
        WHERE related.id <> $entity_id
        WITH related, count(d) as shared_docs
        ORDER BY shared_docs DESC
        LIMIT 20
        RETURN related.value as value, related.entity_type as type, shared_docs
        """

        related_entities = await self._run_query(related_query, {"entity_id": entity_id})

        return {
            "entity": {
                "type": entity_type,
                "value": entity_value,
            },
            "documents": [d["document_id"] for d in documents],
            "document_details": documents,
            "related_entities": related_entities,
        }

    async def get_citation_network(
        self,
        citation: str,
    ) -> dict[str, Any]:
        """
        Get network of documents citing a specific case/statute.

        Args:
            citation: The citation to search for

        Returns:
            Citation network information
        """
        # Find documents with this citation
        query = """
        MATCH (c:Citation)
        WHERE c.normalized CONTAINS $citation OR c.raw_text CONTAINS $citation
        WITH c
        MATCH (d:Document)-[:CITES]->(c)
        RETURN d.id as document_id, d.title as title, c.normalized as citation
        """

        citing_docs = await self._run_query(query, {"citation": citation})

        # Find co-cited citations
        co_cited_query = """
        MATCH (c:Citation)
        WHERE c.normalized CONTAINS $citation OR c.raw_text CONTAINS $citation
        WITH c
        MATCH (d:Document)-[:CITES]->(c)
        MATCH (d)-[:CITES]->(other:Citation)
        WHERE other <> c
        WITH other, count(d) as co_citation_count
        ORDER BY co_citation_count DESC
        LIMIT 10
        RETURN other.normalized as citation, co_citation_count
        """

        co_cited = await self._run_query(co_cited_query, {"citation": citation})

        return {
            "citation": citation,
            "citing_documents": [d["document_id"] for d in citing_docs],
            "document_details": citing_docs,
            "co_cited_with": co_cited,
        }

    async def get_entity_summary(
        self,
        entity_type: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Get summary of entities in the graph.

        Args:
            entity_type: Optional filter by entity type
            limit: Maximum results

        Returns:
            List of entities with document counts
        """
        if entity_type:
            query = f"""
            MATCH (e:{entity_type})<-[:CONTAINS_ENTITY]-(d:Document)
            WITH e, count(d) as doc_count
            ORDER BY doc_count DESC
            LIMIT $limit
            RETURN e.value as value, e.entity_type as type, doc_count
            """
        else:
            query = """
            MATCH (e)<-[:CONTAINS_ENTITY]-(d:Document)
            WHERE e.entity_type IS NOT NULL
            WITH e, count(d) as doc_count
            ORDER BY doc_count DESC
            LIMIT $limit
            RETURN e.value as value, e.entity_type as type, doc_count
            """

        return await self._run_query(query, {"limit": limit})

    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and its relationships from the graph.

        Args:
            document_id: Document ID to delete

        Returns:
            Success status
        """
        query = """
        MATCH (d:Document {id: $doc_id})
        DETACH DELETE d
        RETURN count(*) as deleted
        """

        result = await self._run_query(query, {"doc_id": document_id})
        return len(result) > 0

    async def get_graph_stats(self) -> dict[str, Any]:
        """Get statistics about the knowledge graph."""
        stats_query = """
        MATCH (d:Document)
        WITH count(d) as doc_count
        MATCH (e)
        WHERE e.entity_type IS NOT NULL
        WITH doc_count, count(e) as entity_count
        MATCH (c:Citation)
        WITH doc_count, entity_count, count(c) as citation_count
        MATCH ()-[r]->()
        RETURN doc_count, entity_count, citation_count, count(r) as relationship_count
        """

        results = await self._run_query(stats_query)

        if results:
            return results[0]
        return {
            "doc_count": 0,
            "entity_count": 0,
            "citation_count": 0,
            "relationship_count": 0,
        }
