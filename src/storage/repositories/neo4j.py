"""
Neo4j Repository
================
Async Neo4j repository for knowledge graph operations.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import Neo4jError

from ..config import Neo4jConfig
from ..models import GraphNode, GraphRelationship

logger = logging.getLogger(__name__)


class Neo4jRepository:
    """
    Neo4j repository for knowledge graph operations.

    Manages document relationships, entities, and legal concepts.
    """

    # Node labels
    LABEL_DOCUMENT = "Document"
    LABEL_ENTITY = "Entity"
    LABEL_PERSON = "Person"
    LABEL_ORGANIZATION = "Organization"
    LABEL_CLAUSE = "Clause"
    LABEL_CITATION = "Citation"
    LABEL_MATTER = "Matter"
    LABEL_CLIENT = "Client"
    LABEL_CONCEPT = "LegalConcept"

    # Relationship types
    REL_CONTAINS = "CONTAINS"
    REL_MENTIONS = "MENTIONS"
    REL_REFERENCES = "REFERENCES"
    REL_RELATED_TO = "RELATED_TO"
    REL_BELONGS_TO = "BELONGS_TO"
    REL_SIMILAR_TO = "SIMILAR_TO"
    REL_CITES = "CITES"
    REL_AMENDS = "AMENDS"
    REL_SUPERSEDES = "SUPERSEDES"

    def __init__(self, config: Neo4jConfig):
        self.config = config
        self._driver: Optional[AsyncDriver] = None

    async def connect(self) -> None:
        """Connect to Neo4j."""
        try:
            self._driver = AsyncGraphDatabase.driver(
                self.config.uri,
                auth=(self.config.user, self.config.password),
                database=self.config.database,
                max_connection_lifetime=self.config.max_connection_lifetime,
                max_connection_pool_size=self.config.max_connection_pool_size,
                connection_acquisition_timeout=self.config.connection_acquisition_timeout,
                encrypted=self.config.encrypted,
            )

            # Test connection
            async with self._driver.session() as session:
                await session.run("RETURN 1")

            logger.info(f"Connected to Neo4j at {self.config.uri}")

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    async def disconnect(self) -> None:
        """Close Neo4j connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("Disconnected from Neo4j")

    async def _session(self) -> AsyncSession:
        """Get a session."""
        if not self._driver:
            raise RuntimeError("Neo4j driver not connected")
        return self._driver.session(database=self.config.database)

    async def initialize_schema(self) -> None:
        """Create indexes and constraints."""
        constraints = [
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (d:{self.LABEL_DOCUMENT}) REQUIRE d.id IS UNIQUE",
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (e:{self.LABEL_ENTITY}) REQUIRE e.id IS UNIQUE",
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (m:{self.LABEL_MATTER}) REQUIRE m.id IS UNIQUE",
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (c:{self.LABEL_CLIENT}) REQUIRE c.id IS UNIQUE",
        ]

        indexes = [
            f"CREATE INDEX IF NOT EXISTS FOR (d:{self.LABEL_DOCUMENT}) ON (d.document_type)",
            f"CREATE INDEX IF NOT EXISTS FOR (d:{self.LABEL_DOCUMENT}) ON (d.created_at)",
            f"CREATE INDEX IF NOT EXISTS FOR (e:{self.LABEL_ENTITY}) ON (e.type)",
            f"CREATE INDEX IF NOT EXISTS FOR (e:{self.LABEL_ENTITY}) ON (e.value)",
            f"CREATE FULLTEXT INDEX document_content IF NOT EXISTS FOR (d:Document) ON EACH [d.title, d.content]",
        ]

        async with await self._session() as session:
            for query in constraints + indexes:
                try:
                    await session.run(query)
                except Neo4jError as e:
                    logger.warning(f"Schema creation warning: {e}")

        logger.info("Neo4j schema initialized")

    # Document Node Operations

    async def create_document_node(
        self,
        document_id: str,
        properties: dict[str, Any],
    ) -> GraphNode:
        """Create a document node."""
        async with await self._session() as session:
            result = await session.run(
                f"""
                MERGE (d:{self.LABEL_DOCUMENT} {{id: $id}})
                SET d += $properties
                SET d.updated_at = datetime()
                RETURN d
                """,
                id=document_id,
                properties=properties,
            )
            record = await result.single()

            if record:
                node = record["d"]
                return GraphNode(
                    id=node["id"],
                    label=self.LABEL_DOCUMENT,
                    properties=dict(node),
                )
            raise ValueError(f"Failed to create document node {document_id}")

    async def get_document_node(self, document_id: str) -> Optional[GraphNode]:
        """Get document node by ID."""
        async with await self._session() as session:
            result = await session.run(
                f"""
                MATCH (d:{self.LABEL_DOCUMENT} {{id: $id}})
                RETURN d
                """,
                id=document_id,
            )
            record = await result.single()

            if record:
                node = record["d"]
                return GraphNode(
                    id=node["id"],
                    label=self.LABEL_DOCUMENT,
                    properties=dict(node),
                )
            return None

    async def delete_document_node(self, document_id: str) -> bool:
        """Delete document node and its relationships."""
        async with await self._session() as session:
            result = await session.run(
                f"""
                MATCH (d:{self.LABEL_DOCUMENT} {{id: $id}})
                DETACH DELETE d
                RETURN count(d) as deleted
                """,
                id=document_id,
            )
            record = await result.single()
            return record["deleted"] > 0 if record else False

    # Entity Operations

    async def create_entity_node(
        self,
        entity_id: str,
        entity_type: str,
        value: str,
        properties: Optional[dict[str, Any]] = None,
    ) -> GraphNode:
        """Create or merge an entity node."""
        props = properties or {}
        props["type"] = entity_type
        props["value"] = value

        # Map entity type to label
        label_map = {
            "person": self.LABEL_PERSON,
            "organization": self.LABEL_ORGANIZATION,
            "clause": self.LABEL_CLAUSE,
            "citation": self.LABEL_CITATION,
        }
        labels = [self.LABEL_ENTITY]
        if entity_type.lower() in label_map:
            labels.append(label_map[entity_type.lower()])

        label_str = ":".join(labels)

        async with await self._session() as session:
            result = await session.run(
                f"""
                MERGE (e:{label_str} {{id: $id}})
                SET e += $properties
                SET e.updated_at = datetime()
                RETURN e
                """,
                id=entity_id,
                properties=props,
            )
            record = await result.single()

            if record:
                node = record["e"]
                return GraphNode(
                    id=node["id"],
                    label=label_str,
                    properties=dict(node),
                )
            raise ValueError(f"Failed to create entity node {entity_id}")

    async def link_document_entity(
        self,
        document_id: str,
        entity_id: str,
        relationship_type: str = REL_MENTIONS,
        properties: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Create relationship between document and entity."""
        props = properties or {}
        props["created_at"] = datetime.utcnow().isoformat()

        async with await self._session() as session:
            result = await session.run(
                f"""
                MATCH (d:{self.LABEL_DOCUMENT} {{id: $doc_id}})
                MATCH (e:{self.LABEL_ENTITY} {{id: $entity_id}})
                MERGE (d)-[r:{relationship_type}]->(e)
                SET r += $properties
                RETURN r
                """,
                doc_id=document_id,
                entity_id=entity_id,
                properties=props,
            )
            record = await result.single()
            return record is not None

    # Document Relationships

    async def create_document_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Create relationship between two documents."""
        props = properties or {}
        props["created_at"] = datetime.utcnow().isoformat()

        async with await self._session() as session:
            result = await session.run(
                f"""
                MATCH (s:{self.LABEL_DOCUMENT} {{id: $source_id}})
                MATCH (t:{self.LABEL_DOCUMENT} {{id: $target_id}})
                MERGE (s)-[r:{relationship_type}]->(t)
                SET r += $properties
                RETURN r
                """,
                source_id=source_id,
                target_id=target_id,
                properties=props,
            )
            record = await result.single()
            return record is not None

    async def get_related_documents(
        self,
        document_id: str,
        relationship_type: Optional[str] = None,
        direction: str = "both",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get documents related to a document."""
        rel_pattern = f"[r:{relationship_type}]" if relationship_type else "[r]"

        if direction == "outgoing":
            pattern = f"(d)-{rel_pattern}->(related)"
        elif direction == "incoming":
            pattern = f"(d)<-{rel_pattern}-(related)"
        else:
            pattern = f"(d)-{rel_pattern}-(related)"

        async with await self._session() as session:
            result = await session.run(
                f"""
                MATCH (d:{self.LABEL_DOCUMENT} {{id: $id}})
                MATCH {pattern}
                WHERE related:{self.LABEL_DOCUMENT}
                RETURN related, type(r) as relationship, r as rel_props
                LIMIT $limit
                """,
                id=document_id,
                limit=limit,
            )

            records = await result.data()
            return [
                {
                    "document": dict(r["related"]),
                    "relationship": r["relationship"],
                    "properties": dict(r["rel_props"]) if r["rel_props"] else {},
                }
                for r in records
            ]

    # Matter/Client Organization

    async def create_matter_node(
        self,
        matter_id: str,
        client_id: str,
        properties: Optional[dict[str, Any]] = None,
    ) -> GraphNode:
        """Create or merge a matter node linked to client."""
        props = properties or {}

        async with await self._session() as session:
            result = await session.run(
                f"""
                MERGE (c:{self.LABEL_CLIENT} {{id: $client_id}})
                MERGE (m:{self.LABEL_MATTER} {{id: $matter_id}})
                SET m += $properties
                MERGE (m)-[:{self.REL_BELONGS_TO}]->(c)
                RETURN m
                """,
                matter_id=matter_id,
                client_id=client_id,
                properties=props,
            )
            record = await result.single()

            if record:
                node = record["m"]
                return GraphNode(
                    id=node["id"],
                    label=self.LABEL_MATTER,
                    properties=dict(node),
                )
            raise ValueError(f"Failed to create matter node {matter_id}")

    async def link_document_matter(
        self,
        document_id: str,
        matter_id: str,
    ) -> bool:
        """Link document to matter."""
        async with await self._session() as session:
            result = await session.run(
                f"""
                MATCH (d:{self.LABEL_DOCUMENT} {{id: $doc_id}})
                MATCH (m:{self.LABEL_MATTER} {{id: $matter_id}})
                MERGE (d)-[r:{self.REL_BELONGS_TO}]->(m)
                RETURN r
                """,
                doc_id=document_id,
                matter_id=matter_id,
            )
            record = await result.single()
            return record is not None

    # Graph Queries

    async def find_documents_by_entity(
        self,
        entity_value: str,
        entity_type: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Find documents mentioning an entity."""
        type_filter = "AND e.type = $type" if entity_type else ""

        async with await self._session() as session:
            result = await session.run(
                f"""
                MATCH (d:{self.LABEL_DOCUMENT})-[r]->(e:{self.LABEL_ENTITY})
                WHERE e.value CONTAINS $value {type_filter}
                RETURN d, collect({{entity: e, rel: type(r)}}) as entities
                LIMIT $limit
                """,
                value=entity_value,
                type=entity_type,
                limit=limit,
            )

            records = await result.data()
            return [
                {
                    "document": dict(r["d"]),
                    "entities": [
                        {"entity": dict(e["entity"]), "relationship": e["rel"]}
                        for e in r["entities"]
                    ],
                }
                for r in records
            ]

    async def find_document_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 5,
    ) -> list[dict[str, Any]]:
        """Find path between two documents."""
        async with await self._session() as session:
            result = await session.run(
                f"""
                MATCH path = shortestPath(
                    (s:{self.LABEL_DOCUMENT} {{id: $source}})-[*1..{max_hops}]-(t:{self.LABEL_DOCUMENT} {{id: $target}})
                )
                RETURN [n IN nodes(path) | properties(n)] as nodes,
                       [r IN relationships(path) | type(r)] as relationships
                """,
                source=source_id,
                target=target_id,
            )

            record = await result.single()
            if record:
                return {
                    "nodes": record["nodes"],
                    "relationships": record["relationships"],
                    "length": len(record["relationships"]),
                }
            return None

    async def get_document_context(
        self,
        document_id: str,
        depth: int = 2,
    ) -> dict[str, Any]:
        """Get document with its surrounding context (entities, related docs)."""
        async with await self._session() as session:
            result = await session.run(
                f"""
                MATCH (d:{self.LABEL_DOCUMENT} {{id: $id}})
                OPTIONAL MATCH (d)-[r1]->(e:{self.LABEL_ENTITY})
                OPTIONAL MATCH (d)-[r2]-(related:{self.LABEL_DOCUMENT})
                OPTIONAL MATCH (d)-[r3]->(m:{self.LABEL_MATTER})-[r4]->(c:{self.LABEL_CLIENT})
                RETURN d,
                       collect(DISTINCT {{entity: e, rel: type(r1)}}) as entities,
                       collect(DISTINCT {{doc: related, rel: type(r2)}}) as related_docs,
                       m as matter,
                       c as client
                """,
                id=document_id,
            )

            record = await result.single()
            if not record:
                return None

            return {
                "document": dict(record["d"]) if record["d"] else None,
                "entities": [
                    {"entity": dict(e["entity"]), "relationship": e["rel"]}
                    for e in record["entities"]
                    if e["entity"]
                ],
                "related_documents": [
                    {"document": dict(r["doc"]), "relationship": r["rel"]}
                    for r in record["related_docs"]
                    if r["doc"]
                ],
                "matter": dict(record["matter"]) if record["matter"] else None,
                "client": dict(record["client"]) if record["client"] else None,
            }

    async def find_similar_documents(
        self,
        document_id: str,
        min_shared_entities: int = 3,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Find documents with similar entities."""
        async with await self._session() as session:
            result = await session.run(
                f"""
                MATCH (d:{self.LABEL_DOCUMENT} {{id: $id}})-[:MENTIONS]->(e:{self.LABEL_ENTITY})
                MATCH (similar:{self.LABEL_DOCUMENT})-[:MENTIONS]->(e)
                WHERE similar.id <> $id
                WITH similar, count(e) as shared_entities
                WHERE shared_entities >= $min_shared
                RETURN similar, shared_entities
                ORDER BY shared_entities DESC
                LIMIT $limit
                """,
                id=document_id,
                min_shared=min_shared_entities,
                limit=limit,
            )

            records = await result.data()
            return [
                {
                    "document": dict(r["similar"]),
                    "shared_entities": r["shared_entities"],
                }
                for r in records
            ]

    # Citation Graph

    async def create_citation(
        self,
        citing_doc_id: str,
        citation_text: str,
        cited_case: Optional[str] = None,
        properties: Optional[dict[str, Any]] = None,
    ) -> str:
        """Create citation node and link to document."""
        import hashlib
        citation_id = hashlib.md5(f"{citing_doc_id}:{citation_text}".encode()).hexdigest()

        props = properties or {}
        props["text"] = citation_text
        if cited_case:
            props["cited_case"] = cited_case

        async with await self._session() as session:
            await session.run(
                f"""
                MATCH (d:{self.LABEL_DOCUMENT} {{id: $doc_id}})
                MERGE (c:{self.LABEL_CITATION} {{id: $citation_id}})
                SET c += $properties
                MERGE (d)-[:{self.REL_CITES}]->(c)
                """,
                doc_id=citing_doc_id,
                citation_id=citation_id,
                properties=props,
            )

        return citation_id

    async def get_citation_network(
        self,
        document_id: str,
        depth: int = 2,
    ) -> dict[str, Any]:
        """Get citation network for a document."""
        async with await self._session() as session:
            result = await session.run(
                f"""
                MATCH (d:{self.LABEL_DOCUMENT} {{id: $id}})
                OPTIONAL MATCH (d)-[:CITES]->(cited:{self.LABEL_CITATION})
                OPTIONAL MATCH (citing:{self.LABEL_DOCUMENT})-[:CITES]->(c2:{self.LABEL_CITATION})
                WHERE c2.cited_case = d.title OR c2.text CONTAINS d.id
                RETURN d,
                       collect(DISTINCT cited) as citations,
                       collect(DISTINCT citing) as cited_by
                """,
                id=document_id,
            )

            record = await result.single()
            if not record:
                return None

            return {
                "document": dict(record["d"]) if record["d"] else None,
                "citations": [dict(c) for c in record["citations"] if c],
                "cited_by": [dict(c) for c in record["cited_by"] if c],
            }

    # Statistics

    async def get_statistics(self) -> dict[str, Any]:
        """Get graph statistics."""
        async with await self._session() as session:
            result = await session.run(
                """
                MATCH (n)
                WITH labels(n) as labels, count(n) as count
                UNWIND labels as label
                RETURN label, sum(count) as count
                ORDER BY count DESC
                """
            )
            node_counts = {r["label"]: r["count"] async for r in result}

            result = await session.run(
                """
                MATCH ()-[r]->()
                RETURN type(r) as type, count(r) as count
                ORDER BY count DESC
                """
            )
            rel_counts = {r["type"]: r["count"] async for r in result}

            return {
                "node_counts": node_counts,
                "relationship_counts": rel_counts,
                "total_nodes": sum(node_counts.values()),
                "total_relationships": sum(rel_counts.values()),
            }

    # Health Check

    async def health_check(self) -> dict[str, Any]:
        """Check Neo4j health."""
        if not self._driver:
            return {"status": "disconnected"}

        try:
            async with await self._session() as session:
                result = await session.run(
                    "CALL dbms.components() YIELD name, versions RETURN name, versions"
                )
                record = await result.single()

                stats = await self.get_statistics()

                return {
                    "status": "healthy",
                    "name": record["name"] if record else "unknown",
                    "versions": record["versions"] if record else [],
                    "statistics": stats,
                }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }
