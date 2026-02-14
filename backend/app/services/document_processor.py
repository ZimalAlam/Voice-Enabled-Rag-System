import asyncio
import io
import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
import tempfile
import os
from pathlib import Path
import base64

import fitz  # PyMuPDF
from PIL import Image
import pytesseract
# from pdf2image import convert_from_bytes  # Optional - commented out for minimal setup
import numpy as np
# import cv2  # Optional - commented out for minimal setup
# from unstructured.partition.pdf import partition_pdf  # Optional - commented out for minimal setup
# from unstructured.documents.elements import ElementMetadata  # Optional - commented out for minimal setup
import google.generativeai as genai
import anthropic

from app.config import settings
from app.models.schemas import (
    ProcessedDocument, 
    DocumentChunk, 
    DocumentImage,
    UploadResponse
)

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Service for processing multimodal documents"""
    
    def __init__(self):
        self.genai_client = None
        self.claude_client = None
        self._initialize_ai_clients()
        
        # Image analysis models
        self.chart_detector = None
        self.table_detector = None
        self._initialize_vision_models()
    
    def _initialize_ai_clients(self):
        """Initialize AI clients based on configuration"""
        try:
            if settings.ai_provider == "gemini" and settings.gemini_api_key:
                genai.configure(api_key=settings.gemini_api_key)
                self.genai_client = genai.GenerativeModel(settings.gemini_vision_model)
                logger.info("Gemini client initialized for document processing")
            elif settings.ai_provider == "claude" and settings.claude_api_key:
                self.claude_client = anthropic.AsyncAnthropic(api_key=settings.claude_api_key)
                logger.info("Claude client initialized for document processing")
            else:
                logger.warning("No AI client available for image description")
        except Exception as e:
            logger.error(f"Failed to initialize AI clients: {e}")
    
    def _initialize_vision_models(self):
        """Initialize computer vision models for image analysis"""
        try:
            # Initialize chart detection (using basic CV techniques)
            # In production, you might want to use specialized models
            logger.info("Document processor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize vision models: {e}")
    
    async def process_document(self, file) -> ProcessedDocument:
        """
        Process a multimodal document (PDF)
        
        Args:
            file: UploadFile containing the document
            
        Returns:
            ProcessedDocument with extracted content
        """
        start_time = time.time()
        document_id = str(uuid.uuid4())
        
        try:
            # Read file content
            content = await file.read()
            file_size = len(content)
            
            # Validate file type and process accordingly
            filename_lower = file.filename.lower()
            
            if filename_lower.endswith('.pdf'):
                # Process PDF
                processed_doc = await self._process_pdf(
                    content, 
                    document_id, 
                    file.filename,
                    file_size,
                    start_time
                )
            elif filename_lower.endswith(('.txt', '.md')):
                # Process text file
                processed_doc = await self._process_text_file(
                    content,
                    document_id,
                    file.filename,
                    file_size,
                    start_time
                )
            else:
                raise ValueError("Only PDF, TXT, and MD files are currently supported")
            
            logger.info(f"Document processed: {file.filename} "
                       f"({processed_doc.pages_processed} pages, "
                       f"{len(processed_doc.text_chunks)} chunks, "
                       f"{len(processed_doc.images)} images)")
            
            return processed_doc
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise
    
    async def _process_pdf(self, content: bytes, document_id: str, 
                          filename: str, file_size: int, start_time: float) -> ProcessedDocument:
        """Process a PDF document with multimodal extraction"""
        
        # Initialize result containers
        text_chunks = []
        images = []
        
        # Open PDF with PyMuPDF for detailed analysis
        pdf_document = fitz.open(stream=content, filetype="pdf")
        pages_processed = len(pdf_document)
        
        # Process each page
        for page_num in range(pages_processed):
            page = pdf_document[page_num]
            
            # Extract text with layout information
            page_chunks = await self._extract_text_from_page(
                page, document_id, page_num + 1
            )
            text_chunks.extend(page_chunks)
            
            # Extract images from page
            page_images = await self._extract_images_from_page(
                page, document_id, page_num + 1
            )
            images.extend(page_images)
            
            # Extract tables (if any)
            table_chunks = await self._extract_tables_from_page(
                page, document_id, page_num + 1
            )
            text_chunks.extend(table_chunks)
        
        pdf_document.close()
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        return ProcessedDocument(
            document_id=document_id,
            filename=filename,
            file_type=".pdf",
            file_size=file_size,
            pages_processed=pages_processed,
            text_chunks=text_chunks,
            images=images,
            metadata={
                "processor": "multimodal_pdf_processor",
                "extraction_methods": ["pymupdf"],
                "ocr_enabled": settings.ocr_enabled,
                "images_extracted": len(images)
            },
            processing_time_ms=processing_time
        )
    
    async def _process_text_file(self, content: bytes, document_id: str, 
                               filename: str, file_size: int, start_time: float) -> ProcessedDocument:
        """Process a text file (.txt, .md)"""
        
        try:
            # Decode text content
            text_content = content.decode('utf-8')
            
            # Split into chunks (by paragraphs or lines)
            text_chunks = []
            paragraphs = text_content.split('\n\n')  # Split by double newlines
            
            chunk_index = 0
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # Create text chunk
                chunk = DocumentChunk(
                    id=f"{document_id}_chunk_{chunk_index}",
                    document_id=document_id,
                    content=paragraph,
                    chunk_index=chunk_index,
                    page_number=1,  # Text files are single "page"
                    bbox=None,
                    metadata={
                        "extraction_method": "text_parser",
                        "content_type": "text"
                    }
                )
                text_chunks.append(chunk)
                chunk_index += 1
            
            # If no chunks were created (no double newlines), split by lines
            if not text_chunks:
                lines = text_content.split('\n')
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue
                    
                    chunk = DocumentChunk(
                        id=f"{document_id}_chunk_{i}",
                        document_id=document_id,
                        content=line,
                        chunk_index=i,
                        page_number=1,
                        bbox=None,
                        metadata={
                            "extraction_method": "text_parser",
                            "content_type": "text"
                        }
                    )
                    text_chunks.append(chunk)
            
            # Calculate processing time
            processing_time = int((time.time() - start_time) * 1000)
            
            return ProcessedDocument(
                document_id=document_id,
                filename=filename,
                file_type=filename.split('.')[-1],
                file_size=file_size,
                pages_processed=1,
                text_chunks=text_chunks,
                images=[],  # No images in text files
                metadata={
                    "processor": "text_file_processor",
                    "extraction_methods": ["text_parser"],
                    "encoding": "utf-8",
                    "chunks_created": len(text_chunks)
                },
                processing_time_ms=processing_time
            )
            
        except UnicodeDecodeError:
            raise ValueError(f"Unable to decode text file {filename}. Please ensure it's UTF-8 encoded.")
        except Exception as e:
            logger.error(f"Text file processing failed: {e}")
            raise
    
    async def _extract_text_from_page(self, page, document_id: str, 
                                    page_num: int) -> List[DocumentChunk]:
        """Extract text chunks from a PDF page with layout preservation"""
        chunks = []
        
        try:
            # Method 1: Extract digital text using PyMuPDF
            digital_text_chunks = await self._extract_digital_text(page, document_id, page_num)
            chunks.extend(digital_text_chunks)
            
            # Method 2: Check if page needs OCR/AI vision analysis
            # If digital text is sparse, try AI vision on the entire page
            total_digital_text = sum(len(chunk.content) for chunk in digital_text_chunks)
            
            if total_digital_text < 50:  # Less than 50 characters suggests scanned/image content
                logger.info(f"Page {page_num} has minimal digital text ({total_digital_text} chars), trying AI vision analysis")
                
                # Render page as image and analyze with AI
                ai_text_chunks = await self._extract_text_with_ai_vision(page, document_id, page_num)
                chunks.extend(ai_text_chunks)
            
        except Exception as e:
            logger.error(f"Text extraction failed for page {page_num}: {e}")
        
        return chunks
    
    async def _extract_digital_text(self, page, document_id: str, page_num: int) -> List[DocumentChunk]:
        """Extract digital text using PyMuPDF"""
        chunks = []
        
        try:
            # Get text blocks with layout information
            blocks = page.get_text("dict")
            
            chunk_index = 0
            for block in blocks.get("blocks", []):
                if "lines" not in block:  # Skip image blocks
                    continue
                
                # Combine lines in block
                block_text = ""
                for line in block["lines"]:
                    for span in line["spans"]:
                        block_text += span["text"] + " "
                
                block_text = block_text.strip()
                if not block_text or len(block_text) < 3:  # Skip very short text
                    continue
                
                # Create chunk with bounding box
                bbox = block.get("bbox", [0, 0, 0, 0])
                
                chunk = DocumentChunk(
                    id=f"{document_id}_digital_{page_num}_{chunk_index}",
                    document_id=document_id,
                    content=block_text,
                    chunk_index=chunk_index,
                    page_number=page_num,
                    bbox=list(bbox),
                    metadata={
                        "extraction_method": "pymupdf_digital",
                        "block_type": "text"
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
        
        except Exception as e:
            logger.error(f"Digital text extraction failed for page {page_num}: {e}")
        
        return chunks
    
    async def _extract_text_with_ai_vision(self, page, document_id: str, 
                                          page_num: int) -> List[DocumentChunk]:
        """Extract text using AI vision analysis when OCR fails"""
        chunks = []
        
        try:
            # Only proceed if we have AI vision capability
            if not (self.genai_client or self.claude_client):
                logger.warning("No AI vision available for text extraction")
                return chunks
            
            # Render page as image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale for better quality
            img_data = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data))
            
            # Use AI to extract text from the page image
            extracted_text = await self._extract_text_with_ai(pil_image, page_num)
            
            if extracted_text and len(extracted_text.strip()) > 10:
                # Create a single chunk for AI-extracted text
                chunk = DocumentChunk(
                    id=f"{document_id}_ai_vision_{page_num}_0",
                    document_id=document_id,
                    content=extracted_text.strip(),
                    chunk_index=0,
                    page_number=page_num,
                    bbox=[0, 0, page.rect.width, page.rect.height],  # Full page bbox
                    metadata={
                        "extraction_method": "ai_vision",
                        "ai_provider": settings.ai_provider,
                        "confidence": "medium"
                    }
                )
                chunks.append(chunk)
                logger.info(f"AI vision extracted {len(extracted_text)} characters from page {page_num}")
        
        except Exception as e:
            logger.error(f"AI vision text extraction failed for page {page_num}: {e}")
        
        return chunks
    
    async def _extract_text_with_ai(self, image: Image.Image, page_num: int) -> str:
        """Extract text from image using AI vision"""
        try:
            if settings.ai_provider == "gemini" and self.genai_client:
                return await self._extract_text_with_gemini(image)
            elif settings.ai_provider == "claude" and self.claude_client:
                return await self._extract_text_with_claude(image)
            else:
                return ""
        except Exception as e:
            logger.error(f"AI text extraction failed: {e}")
            return ""
    
    async def _extract_text_with_gemini(self, image: Image.Image) -> str:
        """Extract text using Gemini Vision"""
        try:
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            prompt = """Please extract ALL the text content from this document page. 
            
            Instructions:
            - Extract all visible text exactly as it appears
            - Maintain the logical reading order
            - Include headers, body text, captions, footnotes, etc.
            - If there are tables, preserve the structure
            - Don't add explanations, just return the extracted text
            - If the image contains charts/graphs with text labels, include those too
            
            Extracted text:"""
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.genai_client.generate_content([prompt, {"mime_type": "image/png", "data": img_byte_arr}])
            )
            
            return response.text if response.text else ""
            
        except Exception as e:
            logger.error(f"Gemini text extraction failed: {e}")
            return ""
    
    async def _extract_text_with_claude(self, image: Image.Image) -> str:
        """Extract text using Claude Vision"""
        try:
            # Convert PIL image to base64
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            img_base64 = base64.b64encode(img_byte_arr).decode()
            
            message = await self.claude_client.messages.create(
                model=settings.claude_vision_model,
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": img_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": """Please extract ALL the text content from this document page. 
                                
                                Instructions:
                                - Extract all visible text exactly as it appears
                                - Maintain the logical reading order
                                - Include headers, body text, captions, footnotes, etc.
                                - If there are tables, preserve the structure
                                - Don't add explanations, just return the extracted text
                                - If the image contains charts/graphs with text labels, include those too
                                
                                Extracted text:"""
                            }
                        ]
                    }
                ]
            )
            
            return message.content[0].text if message.content else ""
            
        except Exception as e:
            logger.error(f"Claude text extraction failed: {e}")
            return ""
    
    async def _extract_images_from_page(self, page, document_id: str, 
                                      page_num: int) -> List[DocumentImage]:
        """Extract and analyze images from a PDF page"""
        images = []
        
        try:
            # Get images from page
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Extract image data
                    xref = img[0]
                    base_image = page.parent.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Convert to PIL Image
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    
                    # Analyze image content
                    image_analysis = await self._analyze_image(pil_image, page_num, img_index)
                    
                    # Save image (in production, save to cloud storage)
                    image_id = f"{document_id}_img_{page_num}_{img_index}"
                    image_url = await self._save_image(image_id, image_bytes, image_ext)
                    
                    # Create thumbnail
                    thumbnail_url = await self._create_thumbnail(image_id, pil_image)
                    
                    document_image = DocumentImage(
                        id=image_id,
                        document_id=document_id,
                        page_number=page_num,
                        image_type=image_analysis["type"],
                        caption=image_analysis.get("caption"),
                        description=image_analysis.get("description"),
                        bbox=[0, 0, 0, 0],  # Default bbox
                        image_url=image_url,
                        thumbnail_url=thumbnail_url,
                        metadata={
                            "format": image_ext,
                            "size": list(pil_image.size),
                            "analysis": image_analysis,
                            "extraction_method": "pymupdf"
                        }
                    )
                    images.append(document_image)
                    
                except Exception as e:
                    logger.error(f"Failed to process image {img_index} on page {page_num}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Image extraction failed for page {page_num}: {e}")
        
        return images
    
    async def _extract_tables_from_page(self, page, document_id: str, 
                                      page_num: int) -> List[DocumentChunk]:
        """Extract tables from a PDF page"""
        chunks = []
        
        try:
            # Find tables using layout analysis
            tables = page.find_tables()
            
            for table_index, table in enumerate(tables):
                # Extract table data
                table_data = table.extract()
                
                # Convert table to text representation
                table_text = self._table_to_text(table_data)
                
                if table_text.strip():
                    chunk = DocumentChunk(
                        id=f"{document_id}_table_{page_num}_{table_index}",
                        document_id=document_id,
                        content=table_text,
                        chunk_index=len(chunks),
                        page_number=page_num,
                        bbox=list(table.bbox),
                        metadata={
                            "extraction_method": "pymupdf_tables",
                            "content_type": "table",
                            "rows": len(table_data),
                            "columns": len(table_data[0]) if table_data else 0
                        }
                    )
                    chunks.append(chunk)
        
        except Exception as e:
            logger.error(f"Table extraction failed for page {page_num}: {e}")
        
        return chunks
    
    async def _analyze_image(self, image: Image.Image, page_num: int, 
                           img_index: int) -> Dict[str, Any]:
        """Analyze image content to determine type and generate description"""
        try:
            # Basic image analysis
            analysis = {
                "type": "image",
                "confidence": 0.5
            }
            
            # Convert to array for CV analysis
            img_array = np.array(image.convert('RGB'))
            
            # Detect if it's a chart/graph
            is_chart = await self._detect_chart(img_array)
            if is_chart:
                analysis["type"] = "chart"
                analysis["confidence"] = 0.8
            
            # Generate description using AI if available
            if self.genai_client or self.claude_client:
                try:
                    description = await self._generate_image_description(image)
                    analysis["description"] = description
                except Exception as e:
                    logger.error(f"Description generation failed: {e}")
            
            # Try OCR for text extraction
            if settings.ocr_enabled:
                try:
                    ocr_text = await self._extract_text_from_image(image)
                    if ocr_text.strip():
                        analysis["ocr_text"] = ocr_text
                        analysis["contains_text"] = True
                except Exception as e:
                    logger.error(f"OCR failed: {e}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {"type": "image", "confidence": 0.0}
    
    async def _detect_chart(self, img_array: np.ndarray) -> bool:
        """Detect if image contains charts or graphs"""
        try:
            # Simple heuristic - look for geometric patterns typical in charts
            # In production, use a trained model
            return False
        except Exception as e:
            logger.error(f"Chart detection failed: {e}")
            return False
    
    async def _generate_image_description(self, image: Image.Image) -> str:
        """Generate description of image using AI"""
        if settings.ai_provider == "gemini" and self.genai_client:
            return await self._generate_description_with_gemini(image)
        elif settings.ai_provider == "claude" and self.claude_client:
            return await self._generate_description_with_claude(image)
        else:
            return "AI description not available"
    
    async def _generate_description_with_gemini(self, image: Image.Image) -> str:
        """Generate description using Gemini Vision"""
        try:
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            prompt = "Describe this image in detail, focusing on any text, charts, diagrams, or important visual elements that would be useful for document analysis."
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.genai_client.generate_content([prompt, {"mime_type": "image/png", "data": img_byte_arr}])
            )
            
            return response.text if response.text else "No description generated"
            
        except Exception as e:
            logger.error(f"Gemini description generation failed: {e}")
            return "Description generation failed"
    
    async def _generate_description_with_claude(self, image: Image.Image) -> str:
        """Generate description using Claude Vision"""
        try:
            # Convert PIL image to base64
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            img_base64 = base64.b64encode(img_byte_arr).decode()
            
            message = await self.claude_client.messages.create(
                model=settings.claude_vision_model,
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": img_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": "Describe this image in detail, focusing on any text, charts, diagrams, or important visual elements that would be useful for document analysis."
                            }
                        ]
                    }
                ]
            )
            
            return message.content[0].text if message.content else "No description generated"
            
        except Exception as e:
            logger.error(f"Claude description generation failed: {e}")
            return "Description generation failed"
    
    async def _extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from image using OCR with AI fallback"""
        try:
            # Try OCR first
            text = pytesseract.image_to_string(image)
            if text and len(text.strip()) > 5:
                return text.strip()
        except Exception as e:
            logger.error(f"OCR failed: {e}")
        
        # Fallback to AI vision if OCR fails or returns minimal text
        try:
            ai_text = await self._extract_text_with_ai_vision_only(image)
            return ai_text
        except Exception as e:
            logger.error(f"AI text extraction fallback failed: {e}")
            return ""

    async def _extract_text_with_ai_vision_only(self, image: Image.Image) -> str:
        """Extract text from image using AI vision"""
        try:
            if settings.ai_provider == "gemini" and self.genai_client:
                return await self._extract_text_with_gemini_vision(image)
            elif settings.ai_provider == "claude" and self.claude_client:
                return await self._extract_text_with_claude_vision(image)
            else:
                return ""
        except Exception as e:
            logger.error(f"AI text extraction failed: {e}")
            return ""

    async def _extract_text_with_gemini_vision(self, image: Image.Image) -> str:
        """Extract text using Gemini Vision"""
        try:
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            prompt = """Please extract ALL the text content from this image. 
            
            Instructions:
            - Extract all visible text exactly as it appears
            - Maintain the logical reading order
            - Include all text: headers, body, captions, labels, etc.
            - Don't add explanations or descriptions
            - Just return the extracted text content
            
            Text content:"""
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.genai_client.generate_content([prompt, {"mime_type": "image/png", "data": img_byte_arr}])
            )
            
            return response.text if response.text else ""
            
        except Exception as e:
            logger.error(f"Gemini text extraction failed: {e}")
            return ""

    async def _extract_text_with_claude_vision(self, image: Image.Image) -> str:
        """Extract text using Claude Vision"""
        try:
            # Convert PIL image to base64
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            img_base64 = base64.b64encode(img_byte_arr).decode()
            
            message = await self.claude_client.messages.create(
                model=settings.claude_vision_model,
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": img_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": """Please extract ALL the text content from this image. 
                                
                                Instructions:
                                - Extract all visible text exactly as it appears
                                - Maintain the logical reading order
                                - Include all text: headers, body, captions, labels, etc.
                                - Don't add explanations or descriptions
                                - Just return the extracted text content
                                
                                Text content:"""
                            }
                        ]
                    }
                ]
            )
            
            return message.content[0].text if message.content else ""
            
        except Exception as e:
            logger.error(f"Claude text extraction failed: {e}")
            return ""
    
    async def _save_image(self, image_id: str, image_bytes: bytes, 
                         ext: str) -> str:
        """Save image and return URL"""
        try:
            # In production, save to cloud storage (S3, GCS, etc.)
            # For now, return a placeholder URL
            return f"/images/{image_id}.{ext}"
        except Exception as e:
            logger.error(f"Image saving failed: {e}")
            return ""
    
    async def _create_thumbnail(self, image_id: str, image: Image.Image) -> str:
        """Create and save thumbnail"""
        try:
            # Create thumbnail
            thumbnail = image.copy()
            thumbnail.thumbnail((150, 150), Image.Resampling.LANCZOS)
            
            # In production, save to cloud storage
            return f"/thumbnails/{image_id}_thumb.png"
        except Exception as e:
            logger.error(f"Thumbnail creation failed: {e}")
            return ""
    
    def _table_to_text(self, table_data: List[List[str]]) -> str:
        """Convert table data to text representation"""
        if not table_data:
            return ""
        
        text_lines = []
        for row in table_data:
            row_text = " | ".join(str(cell) if cell else "" for cell in row)
            text_lines.append(row_text)
        
        return "\n".join(text_lines) 