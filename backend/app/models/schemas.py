from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class SourceType(str, Enum):
    """Types of sources for search results"""
    DOCUMENT = "document"
    WEB = "web"
    GOOGLE_DRIVE = "google_drive"
    IMAGE = "image"

class CitationType(str, Enum):
    """Types of citations"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"

# Request Models
class QueryRequest(BaseModel):
    """Request model for querying the system"""
    query: str = Field(..., min_length=1, max_length=1000, description="The user's query")
    num_results: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    include_web_search: bool = Field(default=True, description="Include web search results")
    include_drive_search: bool = Field(default=True, description="Include Google Drive search results")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Additional search filters")

class STTRequest(BaseModel):
    """Request model for speech-to-text"""
    audio_format: str = Field(default="wav", description="Audio format (wav, mp3, m4a)")
    language: Optional[str] = Field(default="en", description="Language code for transcription")
    stream: bool = Field(default=True, description="Whether to use streaming STT")

# Response Models
class Citation(BaseModel):
    """Citation information for a source"""
    id: str = Field(..., description="Unique citation identifier")
    source_type: SourceType = Field(..., description="Type of source")
    citation_type: CitationType = Field(..., description="Type of citation content")
    title: str = Field(..., description="Title of the source")
    content: str = Field(..., description="Relevant content excerpt")
    url: Optional[HttpUrl] = Field(None, description="URL if available")
    page_number: Optional[int] = Field(None, description="Page number in document")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Relevance confidence score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    image_url: Optional[str] = Field(None, description="URL to associated image")
    thumbnail_url: Optional[str] = Field(None, description="URL to thumbnail image")

class SearchResult(BaseModel):
    """Individual search result"""
    id: str = Field(..., description="Unique result identifier")
    title: str = Field(..., description="Result title")
    content: str = Field(..., description="Result content/snippet")
    source_type: SourceType = Field(..., description="Type of source")
    url: Optional[HttpUrl] = Field(None, description="Source URL")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="When result was indexed")

class QueryResponse(BaseModel):
    """Response model for query results"""
    answer: str = Field(..., description="Generated answer to the query")
    citations: List[Citation] = Field(default_factory=list, description="Supporting citations")
    search_results: List[SearchResult] = Field(default_factory=list, description="Raw search results")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in answer")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    sources_used: Dict[SourceType, int] = Field(default_factory=dict, description="Count of sources by type")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional response metadata")

class UploadResponse(BaseModel):
    """Response model for document upload"""
    success: bool = Field(..., description="Whether upload was successful")
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    pages_processed: int = Field(..., description="Number of pages processed")
    images_extracted: int = Field(..., description="Number of images extracted")
    text_chunks: int = Field(..., description="Number of text chunks created")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    message: str = Field(..., description="Status message")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")

class STTResponse(BaseModel):
    """Response model for speech-to-text"""
    text: str = Field(..., description="Transcribed text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Transcription confidence")
    language: str = Field(..., description="Detected language")
    is_partial: bool = Field(default=False, description="Whether this is a partial result")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")

# Document Models
class DocumentChunk(BaseModel):
    """A chunk of processed document content"""
    id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document identifier")
    content: str = Field(..., description="Text content of the chunk")
    chunk_index: int = Field(..., description="Index of chunk in document")
    page_number: Optional[int] = Field(None, description="Page number in original document")
    bbox: Optional[List[float]] = Field(None, description="Bounding box coordinates [x1, y1, x2, y2]")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")

class DocumentImage(BaseModel):
    """An image extracted from a document"""
    id: str = Field(..., description="Unique image identifier")
    document_id: str = Field(..., description="Parent document identifier")
    page_number: int = Field(..., description="Page number where image was found")
    image_type: str = Field(..., description="Type of image (chart, diagram, photo, etc.)")
    caption: Optional[str] = Field(None, description="Image caption if available")
    description: Optional[str] = Field(None, description="AI-generated description")
    bbox: Optional[List[float]] = Field(None, description="Bounding box coordinates [x1, y1, x2, y2]")
    image_url: str = Field(..., description="URL to access the image")
    thumbnail_url: Optional[str] = Field(None, description="URL to thumbnail")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Image metadata")

class ProcessedDocument(BaseModel):
    """A fully processed document"""
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File type/extension")
    file_size: int = Field(..., description="File size in bytes")
    pages_processed: int = Field(..., description="Number of pages processed")
    text_chunks: List[DocumentChunk] = Field(default_factory=list, description="Text chunks")
    images: List[DocumentImage] = Field(default_factory=list, description="Extracted images")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    created_at: datetime = Field(default_factory=datetime.now, description="Processing timestamp")

# Google Drive Models
class DriveFile(BaseModel):
    """Google Drive file information"""
    id: str = Field(..., description="Google Drive file ID")
    name: str = Field(..., description="File name")
    mime_type: str = Field(..., description="MIME type")
    size: Optional[int] = Field(None, description="File size in bytes")
    created_time: datetime = Field(..., description="Creation timestamp")
    modified_time: datetime = Field(..., description="Last modification timestamp")
    web_view_link: Optional[HttpUrl] = Field(None, description="Link to view in browser")
    thumbnail_link: Optional[HttpUrl] = Field(None, description="Thumbnail URL")
    parents: List[str] = Field(default_factory=list, description="Parent folder IDs")

# WebSocket Models
class WebSocketMessage(BaseModel):
    """Base WebSocket message"""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(default_factory=dict, description="Message data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")

class ChatMessage(BaseModel):
    """Chat message for WebSocket communication"""
    message: str = Field(..., description="Chat message content")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: str = Field(..., description="Chat session identifier")

# Error Models
class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

# Status Models
class SystemStatus(BaseModel):
    """System health status"""
    status: str = Field(..., description="Overall system status")
    services: Dict[str, bool] = Field(default_factory=dict, description="Service availability")
    uptime_seconds: int = Field(..., description="System uptime in seconds")
    version: str = Field(..., description="System version")
    timestamp: datetime = Field(default_factory=datetime.now, description="Status timestamp") 