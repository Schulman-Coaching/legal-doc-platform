"""
Document Chunker
=================
Split documents into chunks for embedding and RAG.
"""

import hashlib
import logging
import re
from typing import Optional

from .models import ChunkingStrategy, DocumentChunk

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Service for splitting documents into chunks for processing.
    Supports multiple chunking strategies optimized for legal documents.
    """

    # Legal document section patterns
    SECTION_PATTERNS = [
        r"^(?:ARTICLE|Article|SECTION|Section)\s+[IVXLCDM\d]+",
        r"^\d+\.\d*\s+[A-Z]",
        r"^[A-Z][A-Z\s]+:$",
        r"^(?:WHEREAS|NOW, THEREFORE|IN WITNESS WHEREOF)",
    ]

    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
    ):
        """
        Initialize document chunker.

        Args:
            strategy: Chunking strategy to use
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            min_chunk_size: Minimum chunk size to create
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(
        self,
        document_id: str,
        text: str,
        metadata: Optional[dict] = None,
    ) -> list[DocumentChunk]:
        """
        Split document into chunks.

        Args:
            document_id: Document identifier
            text: Full document text
            metadata: Optional metadata to attach to chunks

        Returns:
            List of document chunks
        """
        if not text or len(text) < self.min_chunk_size:
            return [self._create_chunk(document_id, text, 0, 0, len(text), metadata)]

        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._chunk_fixed_size(document_id, text, metadata)
        elif self.strategy == ChunkingStrategy.SEMANTIC:
            return self._chunk_semantic(document_id, text, metadata)
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            return self._chunk_by_paragraph(document_id, text, metadata)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            return self._chunk_by_sentence(document_id, text, metadata)
        elif self.strategy == ChunkingStrategy.RECURSIVE:
            return self._chunk_recursive(document_id, text, metadata)
        else:
            return self._chunk_fixed_size(document_id, text, metadata)

    def _create_chunk(
        self,
        document_id: str,
        content: str,
        index: int,
        start_char: int,
        end_char: int,
        metadata: Optional[dict] = None,
    ) -> DocumentChunk:
        """Create a document chunk."""
        chunk_id = hashlib.sha256(
            f"{document_id}:{index}:{start_char}".encode()
        ).hexdigest()[:16]

        return DocumentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            content=content.strip(),
            chunk_index=index,
            start_char=start_char,
            end_char=end_char,
            metadata=metadata or {},
        )

    def _chunk_fixed_size(
        self,
        document_id: str,
        text: str,
        metadata: Optional[dict] = None,
    ) -> list[DocumentChunk]:
        """Split text into fixed-size chunks with overlap."""
        chunks = []
        start = 0
        index = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to end at a sentence boundary
            if end < len(text):
                # Look for sentence end within last 20% of chunk
                search_start = end - int(self.chunk_size * 0.2)
                search_text = text[search_start:end + 50]

                # Find last sentence boundary
                for pattern in [r'\.\s+', r'\?\s+', r'!\s+', r'\n\n']:
                    matches = list(re.finditer(pattern, search_text))
                    if matches:
                        last_match = matches[-1]
                        end = search_start + last_match.end()
                        break

            chunk_text = text[start:end]

            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunks.append(self._create_chunk(
                    document_id,
                    chunk_text,
                    index,
                    start,
                    end,
                    metadata,
                ))
                index += 1

            # Move start with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break

        return chunks

    def _chunk_by_paragraph(
        self,
        document_id: str,
        text: str,
        metadata: Optional[dict] = None,
    ) -> list[DocumentChunk]:
        """Split text by paragraphs, merging small ones."""
        # Split by double newlines or indentation patterns
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk = ""
        chunk_start = 0
        index = 0
        current_pos = 0

        for para in paragraphs:
            para_start = text.find(para, current_pos)
            if para_start == -1:
                para_start = current_pos

            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                    chunk_start = para_start
            else:
                # Save current chunk if it meets minimum size
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(self._create_chunk(
                        document_id,
                        current_chunk,
                        index,
                        chunk_start,
                        chunk_start + len(current_chunk),
                        metadata,
                    ))
                    index += 1

                current_chunk = para
                chunk_start = para_start

            current_pos = para_start + len(para)

        # Don't forget last chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(self._create_chunk(
                document_id,
                current_chunk,
                index,
                chunk_start,
                chunk_start + len(current_chunk),
                metadata,
            ))

        return chunks

    def _chunk_by_sentence(
        self,
        document_id: str,
        text: str,
        metadata: Optional[dict] = None,
    ) -> list[DocumentChunk]:
        """Split text by sentences, grouping to target size."""
        # Sentence splitting with legal document awareness
        # Avoid splitting on common abbreviations
        abbrevs = r'(?<!\b(?:Mr|Mrs|Ms|Dr|Prof|Inc|Ltd|Corp|Co|vs|etc|e\.g|i\.e|No|Art|Sec))'
        sentence_pattern = abbrevs + r'[.!?](?:\s+|$)'

        sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""
        chunk_start = 0
        index = 0
        current_pos = 0

        for sentence in sentences:
            sent_start = text.find(sentence, current_pos)
            if sent_start == -1:
                sent_start = current_pos

            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                    chunk_start = sent_start
            else:
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(self._create_chunk(
                        document_id,
                        current_chunk,
                        index,
                        chunk_start,
                        chunk_start + len(current_chunk),
                        metadata,
                    ))
                    index += 1

                current_chunk = sentence
                chunk_start = sent_start

            current_pos = sent_start + len(sentence)

        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(self._create_chunk(
                document_id,
                current_chunk,
                index,
                chunk_start,
                chunk_start + len(current_chunk),
                metadata,
            ))

        return chunks

    def _chunk_semantic(
        self,
        document_id: str,
        text: str,
        metadata: Optional[dict] = None,
    ) -> list[DocumentChunk]:
        """
        Split text by semantic sections (legal document structure).
        Identifies articles, sections, and clauses.
        """
        # Find section boundaries
        section_boundaries = [0]

        for pattern in self.SECTION_PATTERNS:
            for match in re.finditer(pattern, text, re.MULTILINE):
                boundary = match.start()
                if boundary not in section_boundaries:
                    section_boundaries.append(boundary)

        section_boundaries.append(len(text))
        section_boundaries.sort()

        # Create chunks from sections
        chunks = []
        index = 0

        for i in range(len(section_boundaries) - 1):
            start = section_boundaries[i]
            end = section_boundaries[i + 1]
            section_text = text[start:end].strip()

            if not section_text:
                continue

            # If section is too large, recursively chunk it
            if len(section_text) > self.chunk_size:
                sub_chunks = self._chunk_fixed_size(document_id, section_text, metadata)
                for sub_chunk in sub_chunks:
                    sub_chunk.chunk_index = index
                    sub_chunk.start_char += start
                    sub_chunk.end_char += start
                    chunks.append(sub_chunk)
                    index += 1
            elif len(section_text) >= self.min_chunk_size:
                chunks.append(self._create_chunk(
                    document_id,
                    section_text,
                    index,
                    start,
                    end,
                    metadata,
                ))
                index += 1

        # If no sections found, fall back to paragraph chunking
        if not chunks:
            return self._chunk_by_paragraph(document_id, text, metadata)

        return chunks

    def _chunk_recursive(
        self,
        document_id: str,
        text: str,
        metadata: Optional[dict] = None,
    ) -> list[DocumentChunk]:
        """
        Recursive chunking that tries different separators.
        Inspired by LangChain's RecursiveCharacterTextSplitter.
        """
        separators = [
            "\n\n\n",  # Multiple newlines (major section)
            "\n\n",    # Paragraph
            "\n",      # Line
            ". ",      # Sentence
            ", ",      # Clause
            " ",       # Word
            "",        # Character
        ]

        return self._recursive_split(
            document_id,
            text,
            separators,
            metadata,
        )

    def _recursive_split(
        self,
        document_id: str,
        text: str,
        separators: list[str],
        metadata: Optional[dict] = None,
        depth: int = 0,
    ) -> list[DocumentChunk]:
        """Recursively split text using separators."""
        chunks = []

        if len(text) <= self.chunk_size:
            if len(text) >= self.min_chunk_size:
                chunks.append(self._create_chunk(
                    document_id,
                    text,
                    0,
                    0,
                    len(text),
                    metadata,
                ))
            return chunks

        # Find first working separator
        separator = separators[0] if separators else ""
        next_separators = separators[1:] if len(separators) > 1 else [""]

        if separator:
            splits = text.split(separator)
        else:
            # Character-level split
            splits = [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        current_chunk = ""
        chunk_start = 0
        index = 0
        current_pos = 0

        for split in splits:
            if not split:
                continue

            potential_chunk = (
                current_chunk + separator + split
                if current_chunk
                else split
            )

            if len(potential_chunk) <= self.chunk_size:
                if not current_chunk:
                    chunk_start = current_pos
                current_chunk = potential_chunk
            else:
                # Save current chunk
                if current_chunk and len(current_chunk) >= self.min_chunk_size:
                    chunks.append(self._create_chunk(
                        document_id,
                        current_chunk,
                        index,
                        chunk_start,
                        chunk_start + len(current_chunk),
                        metadata,
                    ))
                    index += 1

                # If split is still too large, recurse with finer separators
                if len(split) > self.chunk_size and next_separators:
                    sub_chunks = self._recursive_split(
                        document_id,
                        split,
                        next_separators,
                        metadata,
                        depth + 1,
                    )
                    for sub_chunk in sub_chunks:
                        sub_chunk.chunk_index = index
                        sub_chunk.start_char += current_pos
                        sub_chunk.end_char += current_pos
                        chunks.append(sub_chunk)
                        index += 1
                    current_chunk = ""
                else:
                    current_chunk = split
                    chunk_start = current_pos

            current_pos += len(split) + len(separator)

        # Handle remaining chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(self._create_chunk(
                document_id,
                current_chunk,
                index,
                chunk_start,
                chunk_start + len(current_chunk),
                metadata,
            ))

        # Re-index chunks
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i

        return chunks

    def merge_small_chunks(
        self,
        chunks: list[DocumentChunk],
        min_size: Optional[int] = None,
    ) -> list[DocumentChunk]:
        """Merge chunks that are too small."""
        if not chunks:
            return []

        min_size = min_size or self.min_chunk_size
        merged = []
        current = None

        for chunk in chunks:
            if current is None:
                current = chunk
            elif len(current.content) < min_size:
                # Merge with next chunk
                current = DocumentChunk(
                    chunk_id=current.chunk_id,
                    document_id=current.document_id,
                    content=current.content + "\n\n" + chunk.content,
                    chunk_index=current.chunk_index,
                    start_char=current.start_char,
                    end_char=chunk.end_char,
                    metadata=current.metadata,
                )
            else:
                merged.append(current)
                current = chunk

        if current:
            merged.append(current)

        # Re-index
        for i, chunk in enumerate(merged):
            chunk.chunk_index = i

        return merged
