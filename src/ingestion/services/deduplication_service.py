"""
Document Deduplication Service
==============================
Detects duplicate and near-duplicate documents using multiple strategies:
- Exact hash matching
- Perceptual hashing for images
- Content-based similarity (MinHash/SimHash)
- Metadata-based matching
"""

import asyncio
import hashlib
import logging
import struct
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DuplicateType(str, Enum):
    """Types of duplicate detection."""
    EXACT = "exact"  # Identical hash
    NEAR_DUPLICATE = "near_duplicate"  # Similar content
    METADATA_MATCH = "metadata_match"  # Same metadata
    VISUAL_SIMILAR = "visual_similar"  # Similar images


class DuplicateAction(str, Enum):
    """Actions for duplicate handling."""
    REJECT = "reject"
    LINK = "link"  # Link to original
    VERSION = "version"  # Create new version
    SKIP = "skip"  # Skip silently
    FLAG = "flag"  # Flag for review


@dataclass
class DuplicateResult:
    """Result of duplicate detection."""
    is_duplicate: bool
    duplicate_type: Optional[DuplicateType] = None
    original_document_id: Optional[str] = None
    similarity_score: float = 0.0
    confidence: float = 0.0
    action: DuplicateAction = DuplicateAction.FLAG
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentFingerprint:
    """Fingerprint of a document for deduplication."""
    document_id: str
    # Hash-based fingerprints
    sha256: str
    md5: str
    # Content fingerprints
    simhash: Optional[int] = None
    minhash_signature: Optional[list[int]] = None
    # Perceptual hash for images
    phash: Optional[str] = None
    # Metadata
    file_size: int = 0
    content_length: int = 0  # Text content length
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


class DeduplicationService:
    """
    Service for detecting duplicate documents.

    Strategies:
    1. Exact matching: SHA-256 hash comparison
    2. Near-duplicate: SimHash for text, pHash for images
    3. Content similarity: MinHash with Jaccard similarity
    4. Metadata matching: Filename, size, date combinations
    """

    def __init__(
        self,
        exact_match_threshold: float = 1.0,
        simhash_threshold: int = 3,  # Hamming distance
        minhash_threshold: float = 0.85,  # Jaccard similarity
        phash_threshold: int = 10,  # Hamming distance
        num_perm: int = 128,  # MinHash permutations
    ):
        self.exact_threshold = exact_match_threshold
        self.simhash_threshold = simhash_threshold
        self.minhash_threshold = minhash_threshold
        self.phash_threshold = phash_threshold
        self.num_perm = num_perm

        # In-memory index (in production, use Redis or database)
        self._hash_index: dict[str, str] = {}  # sha256 -> document_id
        self._simhash_index: dict[str, list[tuple[int, str]]] = {}  # bucket -> [(simhash, doc_id)]
        self._minhash_index: dict[int, set[str]] = {}  # band_hash -> set of doc_ids

    async def check_duplicate(
        self,
        content: bytes,
        filename: str,
        metadata: Optional[dict] = None,
    ) -> DuplicateResult:
        """
        Check if document is a duplicate.

        Args:
            content: Document binary content
            filename: Original filename
            metadata: Optional metadata dict

        Returns:
            DuplicateResult with duplicate information
        """
        # Generate fingerprints
        fingerprint = await self._generate_fingerprint(content, filename, metadata)

        # Check exact match first
        result = await self._check_exact_match(fingerprint)
        if result.is_duplicate:
            return result

        # Check near-duplicate
        result = await self._check_near_duplicate(fingerprint)
        if result.is_duplicate:
            return result

        # Check metadata match
        if metadata:
            result = await self._check_metadata_match(fingerprint, metadata)
            if result.is_duplicate:
                return result

        return DuplicateResult(
            is_duplicate=False,
            similarity_score=0.0,
            confidence=1.0,
        )

    async def register_document(
        self,
        document_id: str,
        content: bytes,
        filename: str,
        metadata: Optional[dict] = None,
    ) -> DocumentFingerprint:
        """
        Register a document in the deduplication index.

        Args:
            document_id: Unique document identifier
            content: Document binary content
            filename: Original filename
            metadata: Optional metadata dict

        Returns:
            Generated DocumentFingerprint
        """
        fingerprint = await self._generate_fingerprint(content, filename, metadata)
        fingerprint.document_id = document_id

        # Index by exact hash
        self._hash_index[fingerprint.sha256] = document_id

        # Index by simhash
        if fingerprint.simhash is not None:
            bucket = self._get_simhash_bucket(fingerprint.simhash)
            if bucket not in self._simhash_index:
                self._simhash_index[bucket] = []
            self._simhash_index[bucket].append((fingerprint.simhash, document_id))

        # Index by minhash (LSH banding)
        if fingerprint.minhash_signature:
            for band_hash in self._get_minhash_bands(fingerprint.minhash_signature):
                if band_hash not in self._minhash_index:
                    self._minhash_index[band_hash] = set()
                self._minhash_index[band_hash].add(document_id)

        logger.debug(f"Registered document {document_id} in dedup index")
        return fingerprint

    async def _generate_fingerprint(
        self,
        content: bytes,
        filename: str,
        metadata: Optional[dict] = None,
    ) -> DocumentFingerprint:
        """Generate all fingerprints for a document."""
        loop = asyncio.get_event_loop()

        # Basic hashes
        sha256 = hashlib.sha256(content).hexdigest()
        md5 = hashlib.md5(content).hexdigest()

        fingerprint = DocumentFingerprint(
            document_id="",
            sha256=sha256,
            md5=md5,
            file_size=len(content),
            metadata=metadata or {},
        )

        # Try to extract text and generate content fingerprints
        text_content = await self._extract_text(content, filename)

        if text_content:
            fingerprint.content_length = len(text_content)
            fingerprint.simhash = await loop.run_in_executor(
                None, self._compute_simhash, text_content
            )
            fingerprint.minhash_signature = await loop.run_in_executor(
                None, self._compute_minhash, text_content
            )

        # Generate perceptual hash for images
        if self._is_image(filename):
            fingerprint.phash = await loop.run_in_executor(
                None, self._compute_phash, content
            )

        return fingerprint

    async def _extract_text(self, content: bytes, filename: str) -> Optional[str]:
        """Extract text content from document."""
        # Placeholder - in production, use tika/textract/PyPDF2
        try:
            # Try simple text extraction
            if filename.lower().endswith(".txt"):
                return content.decode("utf-8", errors="ignore")

            # For other formats, would use actual extraction libraries
            return None

        except Exception as e:
            logger.debug(f"Text extraction failed: {e}")
            return None

    def _compute_simhash(self, text: str) -> int:
        """
        Compute SimHash for text content.

        SimHash creates a 64-bit fingerprint where similar texts
        have fingerprints with small Hamming distance.
        """
        # Tokenize
        words = text.lower().split()
        if not words:
            return 0

        # Initialize bit counters
        v = [0] * 64

        # Weight each word
        for word in words:
            # Hash the word
            word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)

            # Update bit counters
            for i in range(64):
                bit = (word_hash >> i) & 1
                v[i] += 1 if bit else -1

        # Generate fingerprint
        fingerprint = 0
        for i in range(64):
            if v[i] > 0:
                fingerprint |= (1 << i)

        return fingerprint

    def _compute_minhash(self, text: str) -> list[int]:
        """
        Compute MinHash signature for text content.

        MinHash approximates Jaccard similarity between document shingles.
        """
        # Create shingles (k-grams)
        k = 5
        words = text.lower().split()
        shingles = set()

        for i in range(len(words) - k + 1):
            shingle = " ".join(words[i:i + k])
            shingles.add(shingle)

        if not shingles:
            return []

        # Generate hash functions and compute signature
        signature = []
        for i in range(self.num_perm):
            min_hash = float("inf")
            for shingle in shingles:
                # Use different hash for each permutation
                h = int(hashlib.md5(f"{i}:{shingle}".encode()).hexdigest(), 16)
                min_hash = min(min_hash, h)
            signature.append(min_hash)

        return signature

    def _compute_phash(self, content: bytes) -> Optional[str]:
        """
        Compute perceptual hash for images.

        Uses DCT-based hashing for image similarity.
        """
        try:
            from PIL import Image
            import io

            # Load image
            img = Image.open(io.BytesIO(content))

            # Resize to 32x32 and convert to grayscale
            img = img.resize((32, 32)).convert("L")

            # Get pixels
            pixels = list(img.getdata())

            # Calculate mean
            avg = sum(pixels) / len(pixels)

            # Generate hash
            bits = "".join("1" if p > avg else "0" for p in pixels)

            # Convert to hex
            return hex(int(bits, 2))[2:].zfill(256)

        except Exception as e:
            logger.debug(f"pHash computation failed: {e}")
            return None

    def _is_image(self, filename: str) -> bool:
        """Check if file is an image based on extension."""
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif"}
        return Path(filename).suffix.lower() in image_extensions

    async def _check_exact_match(self, fingerprint: DocumentFingerprint) -> DuplicateResult:
        """Check for exact hash match."""
        if fingerprint.sha256 in self._hash_index:
            original_id = self._hash_index[fingerprint.sha256]
            return DuplicateResult(
                is_duplicate=True,
                duplicate_type=DuplicateType.EXACT,
                original_document_id=original_id,
                similarity_score=1.0,
                confidence=1.0,
                action=DuplicateAction.REJECT,
                details={"match_type": "sha256"},
            )

        return DuplicateResult(is_duplicate=False)

    async def _check_near_duplicate(self, fingerprint: DocumentFingerprint) -> DuplicateResult:
        """Check for near-duplicate using SimHash and MinHash."""
        # Check SimHash
        if fingerprint.simhash is not None:
            bucket = self._get_simhash_bucket(fingerprint.simhash)

            if bucket in self._simhash_index:
                for stored_hash, doc_id in self._simhash_index[bucket]:
                    distance = self._hamming_distance(fingerprint.simhash, stored_hash)

                    if distance <= self.simhash_threshold:
                        similarity = 1.0 - (distance / 64.0)
                        return DuplicateResult(
                            is_duplicate=True,
                            duplicate_type=DuplicateType.NEAR_DUPLICATE,
                            original_document_id=doc_id,
                            similarity_score=similarity,
                            confidence=0.9,
                            action=DuplicateAction.FLAG,
                            details={
                                "match_type": "simhash",
                                "hamming_distance": distance,
                            },
                        )

        # Check MinHash
        if fingerprint.minhash_signature:
            candidate_ids = set()
            for band_hash in self._get_minhash_bands(fingerprint.minhash_signature):
                if band_hash in self._minhash_index:
                    candidate_ids.update(self._minhash_index[band_hash])

            # For candidates, compute actual Jaccard similarity
            for doc_id in candidate_ids:
                # Would need to retrieve stored signature to compute
                # For now, flag as potential duplicate
                return DuplicateResult(
                    is_duplicate=True,
                    duplicate_type=DuplicateType.NEAR_DUPLICATE,
                    original_document_id=doc_id,
                    similarity_score=0.85,
                    confidence=0.8,
                    action=DuplicateAction.FLAG,
                    details={"match_type": "minhash"},
                )

        return DuplicateResult(is_duplicate=False)

    async def _check_metadata_match(
        self,
        fingerprint: DocumentFingerprint,
        metadata: dict,
    ) -> DuplicateResult:
        """Check for matches based on metadata."""
        # This would query a database in production
        # Checking for same filename + same size + same date

        return DuplicateResult(is_duplicate=False)

    def _get_simhash_bucket(self, simhash: int) -> str:
        """Get bucket key for SimHash (first 16 bits)."""
        return f"sh_{(simhash >> 48) & 0xFFFF}"

    def _get_minhash_bands(self, signature: list[int]) -> list[int]:
        """Get band hashes for LSH."""
        bands = []
        band_size = len(signature) // 16  # 16 bands

        for i in range(16):
            band = signature[i * band_size:(i + 1) * band_size]
            band_hash = hash(tuple(band))
            bands.append(band_hash)

        return bands

    def _hamming_distance(self, hash1: int, hash2: int) -> int:
        """Compute Hamming distance between two hashes."""
        xor = hash1 ^ hash2
        distance = 0
        while xor:
            distance += xor & 1
            xor >>= 1
        return distance

    def get_stats(self) -> dict:
        """Get deduplication index statistics."""
        return {
            "exact_hash_count": len(self._hash_index),
            "simhash_buckets": len(self._simhash_index),
            "minhash_bands": len(self._minhash_index),
        }

    async def clear_index(self) -> None:
        """Clear all indexes."""
        self._hash_index.clear()
        self._simhash_index.clear()
        self._minhash_index.clear()
        logger.info("Deduplication index cleared")
