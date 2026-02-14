import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
import json
import io
from datetime import datetime, timedelta

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
import httpx

from app.config import settings
from app.models.schemas import SearchResult, DriveFile, SourceType

logger = logging.getLogger(__name__)

class GoogleDriveService:
    """Service for accessing Google Drive with MCP support"""
    
    # Google Drive API scopes
    SCOPES = [
        'https://www.googleapis.com/auth/drive.readonly',
        'https://www.googleapis.com/auth/drive.metadata.readonly'
    ]
    
    def __init__(self):
        self.credentials = None
        self.drive_service = None
        self.docs_service = None
        self.sheets_service = None
        self.slides_service = None
        self._file_cache = {}  # Cache for file metadata
        self._content_cache = {}  # Cache for file content
        
    async def initialize(self):
        """Initialize Google Drive API services"""
        try:
            await self._setup_credentials()
            if self.credentials:
                self._build_services()
                logger.info("Google Drive service initialized successfully")
            else:
                logger.warning("Google Drive service not initialized - no credentials")
        except Exception as e:
            logger.error(f"Failed to initialize Google Drive service: {e}")
    
    async def _setup_credentials(self):
        """Setup Google API credentials"""
        if not settings.google_drive_client_id or not settings.google_drive_client_secret:
            logger.warning("Google Drive credentials not configured")
            return
        
        # In production, implement proper OAuth flow and token storage
        # For now, this is a placeholder for credential management
        try:
            # Load existing credentials from storage (implement this)
            creds_data = await self._load_stored_credentials()
            
            if creds_data:
                self.credentials = Credentials.from_authorized_user_info(
                    creds_data, self.SCOPES
                )
                
                # Refresh if expired
                if self.credentials.expired and self.credentials.refresh_token:
                    self.credentials.refresh(Request())
                    await self._store_credentials(self.credentials)
            else:
                logger.info("No stored credentials found. OAuth flow required.")
                
        except Exception as e:
            logger.error(f"Credential setup failed: {e}")
    
    def _build_services(self):
        """Build Google API service objects"""
        if not self.credentials:
            return
        
        try:
            self.drive_service = build('drive', 'v3', credentials=self.credentials)
            self.docs_service = build('docs', 'v1', credentials=self.credentials)
            self.sheets_service = build('sheets', 'v4', credentials=self.credentials)
            self.slides_service = build('slides', 'v1', credentials=self.credentials)
        except Exception as e:
            logger.error(f"Failed to build Google API services: {e}")
    
    async def search(self, query: str, num_results: int = 5, 
                    filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Search Google Drive for files and content
        
        Args:
            query: Search query
            num_results: Number of results to return
            filters: Additional search filters
            
        Returns:
            List of SearchResult objects
        """
        if not self.drive_service:
            logger.warning("Google Drive service not available")
            return []
        
        start_time = time.time()
        results = []
        
        try:
            # Build search query for Google Drive API
            drive_query = self._build_drive_query(query, filters)
            
            # Search for files
            files = await self._search_files(drive_query, num_results * 2)  # Get more files to filter
            
            # Process and rank results
            for file_data in files[:num_results]:
                search_result = await self._convert_file_to_search_result(file_data, query)
                if search_result:
                    results.append(search_result)
            
            processing_time = int((time.time() - start_time) * 1000)
            logger.info(f"Google Drive search completed in {processing_time}ms: {len(results)} results")
            
        except Exception as e:
            logger.error(f"Google Drive search failed: {e}")
        
        return results
    
    def _build_drive_query(self, query: str, filters: Optional[Dict[str, Any]] = None) -> str:
        """Build Google Drive API search query"""
        # Base query for content search
        query_parts = [f"fullText contains '{query}'"]
        
        # Add file type filters
        if filters:
            if "file_types" in filters:
                type_queries = []
                for file_type in filters["file_types"]:
                    if file_type == "document":
                        type_queries.append("mimeType='application/vnd.google-apps.document'")
                    elif file_type == "spreadsheet":
                        type_queries.append("mimeType='application/vnd.google-apps.spreadsheet'")
                    elif file_type == "presentation":
                        type_queries.append("mimeType='application/vnd.google-apps.presentation'")
                    elif file_type == "pdf":
                        type_queries.append("mimeType='application/pdf'")
                
                if type_queries:
                    query_parts.append(f"({' or '.join(type_queries)})")
            
            # Add date filters
            if "modified_after" in filters:
                date_str = filters["modified_after"]
                query_parts.append(f"modifiedTime > '{date_str}'")
        
        # Exclude trashed files
        query_parts.append("trashed=false")
        
        return " and ".join(query_parts)
    
    async def _search_files(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search for files using Google Drive API"""
        try:
            # Execute search request
            results = self.drive_service.files().list(
                q=query,
                pageSize=min(max_results, 100),
                fields="nextPageToken, files(id, name, mimeType, size, createdTime, modifiedTime, webViewLink, thumbnailLink, parents, description)",
                orderBy="relevance desc"
            ).execute()
            
            files = results.get('files', [])
            
            # Get additional pages if needed
            while 'nextPageToken' in results and len(files) < max_results:
                results = self.drive_service.files().list_next(
                    previous_request=self.drive_service.files().list(q=query),
                    previous_response=results
                ).execute()
                files.extend(results.get('files', []))
            
            return files[:max_results]
            
        except HttpError as e:
            logger.error(f"Google Drive API error: {e}")
            return []
    
    async def _convert_file_to_search_result(self, file_data: Dict[str, Any], 
                                           query: str) -> Optional[SearchResult]:
        """Convert Google Drive file to SearchResult"""
        try:
            file_id = file_data['id']
            file_name = file_data['name']
            mime_type = file_data['mimeType']
            
            # Get file content snippet
            content_snippet = await self._get_file_content_snippet(file_id, mime_type, query)
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(file_data, query, content_snippet)
            
            search_result = SearchResult(
                id=f"gdrive_{file_id}",
                title=file_name,
                content=content_snippet,
                source_type=SourceType.GOOGLE_DRIVE,
                url=file_data.get('webViewLink'),
                confidence_score=relevance_score,
                metadata={
                    "file_id": file_id,
                    "mime_type": mime_type,
                    "file_size": file_data.get('size'),
                    "created_time": file_data.get('createdTime'),
                    "modified_time": file_data.get('modifiedTime'),
                    "thumbnail_link": file_data.get('thumbnailLink'),
                    "parents": file_data.get('parents', []),
                    "description": file_data.get('description', ''),
                    "service": "google_drive"
                }
            )
            
            return search_result
            
        except Exception as e:
            logger.error(f"Failed to convert file to search result: {e}")
            return None
    
    async def _get_file_content_snippet(self, file_id: str, mime_type: str, 
                                      query: str) -> str:
        """Extract content snippet from Google Drive file"""
        try:
            # Check cache first
            cache_key = f"{file_id}_{query}"
            if cache_key in self._content_cache:
                return self._content_cache[cache_key]
            
            content = ""
            
            if mime_type == 'application/vnd.google-apps.document':
                content = await self._get_document_content(file_id, query)
            elif mime_type == 'application/vnd.google-apps.spreadsheet':
                content = await self._get_spreadsheet_content(file_id, query)
            elif mime_type == 'application/vnd.google-apps.presentation':
                content = await self._get_presentation_content(file_id, query)
            elif mime_type == 'application/pdf':
                content = await self._get_pdf_content(file_id, query)
            else:
                # Try to export as text
                content = await self._export_file_as_text(file_id, mime_type)
            
            # Create snippet around query terms
            snippet = self._create_content_snippet(content, query)
            
            # Cache the result
            self._content_cache[cache_key] = snippet
            
            return snippet
            
        except Exception as e:
            logger.error(f"Failed to get content snippet for {file_id}: {e}")
            return ""
    
    async def _get_document_content(self, file_id: str, query: str) -> str:
        """Get content from Google Docs document"""
        try:
            document = self.docs_service.documents().get(documentId=file_id).execute()
            
            content = ""
            for element in document.get('body', {}).get('content', []):
                if 'paragraph' in element:
                    paragraph = element['paragraph']
                    for text_element in paragraph.get('elements', []):
                        if 'textRun' in text_element:
                            content += text_element['textRun'].get('content', '')
            
            return content
            
        except HttpError as e:
            logger.error(f"Failed to get document content: {e}")
            return ""
    
    async def _get_spreadsheet_content(self, file_id: str, query: str) -> str:
        """Get content from Google Sheets spreadsheet"""
        try:
            spreadsheet = self.sheets_service.spreadsheets().get(
                spreadsheetId=file_id
            ).execute()
            
            content_parts = []
            
            # Get all sheet data
            for sheet in spreadsheet.get('sheets', []):
                sheet_title = sheet['properties']['title']
                
                # Get values from sheet
                range_name = f"'{sheet_title}'"
                result = self.sheets_service.spreadsheets().values().get(
                    spreadsheetId=file_id,
                    range=range_name
                ).execute()
                
                values = result.get('values', [])
                
                # Convert to text
                sheet_content = f"\n--- Sheet: {sheet_title} ---\n"
                for row in values:
                    sheet_content += " | ".join(str(cell) for cell in row) + "\n"
                
                content_parts.append(sheet_content)
            
            return "\n".join(content_parts)
            
        except HttpError as e:
            logger.error(f"Failed to get spreadsheet content: {e}")
            return ""
    
    async def _get_presentation_content(self, file_id: str, query: str) -> str:
        """Get content from Google Slides presentation"""
        try:
            presentation = self.slides_service.presentations().get(
                presentationId=file_id
            ).execute()
            
            content_parts = []
            
            for i, slide in enumerate(presentation.get('slides', [])):
                slide_content = f"\n--- Slide {i+1} ---\n"
                
                for element in slide.get('pageElements', []):
                    if 'shape' in element:
                        shape = element['shape']
                        if 'text' in shape:
                            text_elements = shape['text'].get('textElements', [])
                            for text_element in text_elements:
                                if 'textRun' in text_element:
                                    slide_content += text_element['textRun'].get('content', '')
                
                content_parts.append(slide_content)
            
            return "\n".join(content_parts)
            
        except HttpError as e:
            logger.error(f"Failed to get presentation content: {e}")
            return ""
    
    async def _get_pdf_content(self, file_id: str, query: str) -> str:
        """Get content from PDF file (basic text extraction)"""
        try:
            # Download PDF file
            request = self.drive_service.files().get_media(fileId=file_id)
            file_content = io.BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            # Basic PDF text extraction would go here
            # For now, return a placeholder
            return "PDF content extraction not implemented"
            
        except HttpError as e:
            logger.error(f"Failed to get PDF content: {e}")
            return ""
    
    async def _export_file_as_text(self, file_id: str, mime_type: str) -> str:
        """Export file as plain text"""
        try:
            request = self.drive_service.files().export_media(
                fileId=file_id,
                mimeType='text/plain'
            )
            
            file_content = io.BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            return file_content.getvalue().decode('utf-8')
            
        except HttpError as e:
            logger.debug(f"Failed to export file as text: {e}")
            return ""
    
    def _create_content_snippet(self, content: str, query: str, 
                              snippet_length: int = 300) -> str:
        """Create a content snippet around query terms"""
        if not content:
            return ""
        
        # Find query terms in content (case insensitive)
        query_terms = query.lower().split()
        content_lower = content.lower()
        
        best_position = 0
        best_score = 0
        
        # Find best position for snippet
        for i in range(len(content) - snippet_length):
            snippet = content_lower[i:i + snippet_length]
            score = sum(1 for term in query_terms if term in snippet)
            
            if score > best_score:
                best_score = score
                best_position = i
        
        # Extract snippet
        snippet_start = max(0, best_position)
        snippet_end = min(len(content), snippet_start + snippet_length)
        snippet = content[snippet_start:snippet_end].strip()
        
        # Add ellipsis if needed
        if snippet_start > 0:
            snippet = "..." + snippet
        if snippet_end < len(content):
            snippet = snippet + "..."
        
        return snippet
    
    def _calculate_relevance_score(self, file_data: Dict[str, Any], 
                                 query: str, content: str) -> float:
        """Calculate relevance score for search result"""
        score = 0.0
        
        # Title match
        title = file_data.get('name', '').lower()
        query_lower = query.lower()
        
        if query_lower in title:
            score += 0.3
        
        # Content match
        if content:
            content_lower = content.lower()
            query_terms = query_lower.split()
            
            term_matches = sum(1 for term in query_terms if term in content_lower)
            score += (term_matches / len(query_terms)) * 0.5
        
        # File type boost
        mime_type = file_data.get('mimeType', '')
        if 'google-apps' in mime_type:
            score += 0.1  # Boost for native Google files
        
        # Recency boost
        modified_time = file_data.get('modifiedTime')
        if modified_time:
            try:
                modified_date = datetime.fromisoformat(modified_time.replace('Z', '+00:00'))
                days_old = (datetime.now().replace(tzinfo=modified_date.tzinfo) - modified_date).days
                
                if days_old < 30:
                    score += 0.1
            except:
                pass
        
        return min(score, 1.0)
    
    async def get_file_content(self, file_id: str) -> Dict[str, Any]:
        """Get full content of a specific file"""
        if not self.drive_service:
            raise Exception("Google Drive service not available")
        
        try:
            # Get file metadata
            file_data = self.drive_service.files().get(
                fileId=file_id,
                fields="id, name, mimeType, size, createdTime, modifiedTime, webViewLink, thumbnailLink"
            ).execute()
            
            mime_type = file_data['mimeType']
            
            # Get full content based on type
            if mime_type == 'application/vnd.google-apps.document':
                content = await self._get_document_content(file_id, "")
            elif mime_type == 'application/vnd.google-apps.spreadsheet':
                content = await self._get_spreadsheet_content(file_id, "")
            elif mime_type == 'application/vnd.google-apps.presentation':
                content = await self._get_presentation_content(file_id, "")
            else:
                content = await self._export_file_as_text(file_id, mime_type)
            
            return {
                "file_data": file_data,
                "content": content
            }
            
        except HttpError as e:
            logger.error(f"Failed to get file content: {e}")
            raise
    
    async def _load_stored_credentials(self) -> Optional[Dict[str, Any]]:
        """Load stored credentials (implement based on your storage solution)"""
        # Placeholder for credential storage implementation
        # In production, load from secure storage (database, file, etc.)
        return None
    
    async def _store_credentials(self, credentials: Credentials):
        """Store credentials securely (implement based on your storage solution)"""
        # Placeholder for credential storage implementation
        # In production, store securely (database, encrypted file, etc.)
        pass
    
    def is_available(self) -> bool:
        """Check if Google Drive service is available"""
        return self.drive_service is not None
    
    async def cleanup(self):
        """Cleanup resources"""
        self._file_cache.clear()
        self._content_cache.clear() 