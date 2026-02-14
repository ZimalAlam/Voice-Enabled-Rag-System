import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
import aiohttp
import json
from urllib.parse import quote_plus, urljoin
from bs4 import BeautifulSoup
import re

from app.config import settings
from app.models.schemas import SearchResult, SourceType

logger = logging.getLogger(__name__)

class WebSearchService:
    """Service for searching the web and retrieving real-time information"""
    
    def __init__(self):
        self.session = None
        self.search_engines = {
            "serp": self._search_serp_api,
            "google": self._search_google_api,
            "bing": self._search_bing_api,
            "fallback": self._search_fallback
        }
        self.content_cache = {}  # Simple in-memory cache
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def search(self, query: str, num_results: int = 5, 
                    filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Search the web for information
        
        Args:
            query: Search query
            num_results: Number of results to return
            filters: Additional search filters
            
        Returns:
            List of SearchResult objects
        """
        start_time = time.time()
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Try search engines in order of preference
            results = []
            
            # Primary: SERP API (Google)
            if settings.serp_api_key:
                try:
                    results = await self._search_serp_api(query, num_results, filters)
                    if results:
                        logger.info(f"SERP API search successful: {len(results)} results")
                except Exception as e:
                    logger.error(f"SERP API search failed: {e}")
            
            # Fallback: Custom Google search
            if not results and settings.google_api_key:
                try:
                    results = await self._search_google_api(query, num_results, filters)
                    if results:
                        logger.info(f"Google API search successful: {len(results)} results")
                except Exception as e:
                    logger.error(f"Google API search failed: {e}")
            
            # Last resort: Scraping-based fallback
            if not results:
                try:
                    results = await self._search_fallback(query, num_results, filters)
                    if results:
                        logger.info(f"Fallback search successful: {len(results)} results")
                except Exception as e:
                    logger.error(f"Fallback search failed: {e}")
            
            # Enhance results with content extraction
            enhanced_results = await self._enhance_search_results(results)
            
            processing_time = int((time.time() - start_time) * 1000)
            logger.info(f"Web search completed in {processing_time}ms: {len(enhanced_results)} results")
            
            return enhanced_results[:num_results]
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []
    
    async def _search_serp_api(self, query: str, num_results: int, 
                              filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search using SERP API (serpapi.com)"""
        url = "https://serpapi.com/search"
        
        params = {
            "q": query,
            "api_key": settings.serp_api_key,
            "engine": "google",
            "num": min(num_results, 20),
            "hl": "en",
            "gl": "us"
        }
        
        # Add filters if provided
        if filters:
            if "time_range" in filters:
                params["tbs"] = f"qdr:{filters['time_range']}"
            if "site" in filters:
                params["q"] = f"site:{filters['site']} {query}"
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return self._parse_serp_results(data)
            else:
                logger.error(f"SERP API error: {response.status}")
                return []
    
    async def _search_google_api(self, query: str, num_results: int,
                                filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search using Google Custom Search API"""
        url = "https://www.googleapis.com/customsearch/v1"
        
        params = {
            "key": settings.google_api_key,
            "cx": "your_custom_search_engine_id",  # Configure this
            "q": query,
            "num": min(num_results, 10)
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return self._parse_google_results(data)
            else:
                logger.error(f"Google API error: {response.status}")
                return []
    
    async def _search_bing_api(self, query: str, num_results: int,
                              filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search using Bing Search API"""
        url = "https://api.bing.microsoft.com/v7.0/search"
        
        headers = {
            "Ocp-Apim-Subscription-Key": "your_bing_api_key"  # Configure this
        }
        
        params = {
            "q": query,
            "count": min(num_results, 20),
            "mkt": "en-US"
        }
        
        async with self.session.get(url, params=params, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return self._parse_bing_results(data)
            else:
                logger.error(f"Bing API error: {response.status}")
                return []
    
    async def _search_fallback(self, query: str, num_results: int,
                              filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Fallback search using DuckDuckGo or direct scraping"""
        try:
            # Use DuckDuckGo as fallback (no API key required)
            url = "https://html.duckduckgo.com/html/"
            params = {"q": query}
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    return self._parse_duckduckgo_results(html, num_results)
                else:
                    return []
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []
    
    def _parse_serp_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse SERP API response"""
        results = []
        
        organic_results = data.get("organic_results", [])
        for idx, result in enumerate(organic_results):
            search_result = SearchResult(
                id=f"serp_{idx}_{hash(result.get('link', ''))}",
                title=result.get("title", ""),
                content=result.get("snippet", ""),
                source_type=SourceType.WEB,
                url=result.get("link"),
                confidence_score=0.9,  # SERP API results are generally high quality
                metadata={
                    "search_engine": "google_serp",
                    "position": result.get("position", idx + 1),
                    "displayed_link": result.get("displayed_link", ""),
                    "rich_snippet": result.get("rich_snippet", {})
                }
            )
            results.append(search_result)
        
        return results
    
    def _parse_google_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse Google Custom Search API response"""
        results = []
        
        items = data.get("items", [])
        for idx, item in enumerate(items):
            search_result = SearchResult(
                id=f"google_{idx}_{hash(item.get('link', ''))}",
                title=item.get("title", ""),
                content=item.get("snippet", ""),
                source_type=SourceType.WEB,
                url=item.get("link"),
                confidence_score=0.85,
                metadata={
                    "search_engine": "google_custom",
                    "formatted_url": item.get("formattedUrl", ""),
                    "html_snippet": item.get("htmlSnippet", "")
                }
            )
            results.append(search_result)
        
        return results
    
    def _parse_bing_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse Bing Search API response"""
        results = []
        
        web_pages = data.get("webPages", {}).get("value", [])
        for idx, page in enumerate(web_pages):
            search_result = SearchResult(
                id=f"bing_{idx}_{hash(page.get('url', ''))}",
                title=page.get("name", ""),
                content=page.get("snippet", ""),
                source_type=SourceType.WEB,
                url=page.get("url"),
                confidence_score=0.8,
                metadata={
                    "search_engine": "bing",
                    "display_url": page.get("displayUrl", ""),
                    "deep_links": page.get("deepLinks", [])
                }
            )
            results.append(search_result)
        
        return results
    
    def _parse_duckduckgo_results(self, html: str, num_results: int) -> List[SearchResult]:
        """Parse DuckDuckGo HTML response"""
        results = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            result_divs = soup.find_all('div', class_='result')
            
            for idx, div in enumerate(result_divs[:num_results]):
                title_element = div.find('a', class_='result__a')
                snippet_element = div.find('a', class_='result__snippet')
                
                if title_element and snippet_element:
                    title = title_element.get_text(strip=True)
                    url = title_element.get('href')
                    snippet = snippet_element.get_text(strip=True)
                    
                    search_result = SearchResult(
                        id=f"ddg_{idx}_{hash(url or '')}",
                        title=title,
                        content=snippet,
                        source_type=SourceType.WEB,
                        url=url,
                        confidence_score=0.7,
                        metadata={
                            "search_engine": "duckduckgo",
                            "extraction_method": "html_parsing"
                        }
                    )
                    results.append(search_result)
        
        except Exception as e:
            logger.error(f"Failed to parse DuckDuckGo results: {e}")
        
        return results
    
    async def _enhance_search_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Enhance search results by extracting more content from pages"""
        enhanced_results = []
        
        # Process results in parallel for better performance
        tasks = []
        for result in results:
            tasks.append(self._extract_page_content(result))
        
        if tasks:
            enhanced = await asyncio.gather(*tasks, return_exceptions=True)
            
            for enhanced_result in enhanced:
                if isinstance(enhanced_result, SearchResult):
                    enhanced_results.append(enhanced_result)
                elif isinstance(enhanced_result, Exception):
                    logger.error(f"Content extraction failed: {enhanced_result}")
        
        return enhanced_results
    
    async def _extract_page_content(self, result: SearchResult) -> SearchResult:
        """Extract additional content from a web page"""
        if not result.url or result.url in self.content_cache:
            if result.url in self.content_cache:
                result.content = self.content_cache[result.url]
            return result
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            async with self.session.get(
                result.url, 
                headers=headers, 
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    html = await response.text()
                    content = self._extract_text_from_html(html)
                    
                    # Cache the content
                    self.content_cache[result.url] = content
                    
                    # Update result with enhanced content
                    if content and len(content) > len(result.content):
                        result.content = content[:1000] + "..." if len(content) > 1000 else content
                        result.confidence_score = min(result.confidence_score + 0.1, 1.0)
        
        except Exception as e:
            logger.debug(f"Failed to extract content from {result.url}: {e}")
        
        return result
    
    def _extract_text_from_html(self, html: str) -> str:
        """Extract clean text content from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Try to find main content areas
            content_selectors = [
                'main', 'article', '.content', '.main-content', 
                '.post-content', '.entry-content', '#content'
            ]
            
            content_text = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content_text = elements[0].get_text(separator=' ', strip=True)
                    break
            
            # Fallback to body text
            if not content_text:
                body = soup.find('body')
                if body:
                    content_text = body.get_text(separator=' ', strip=True)
            
            # Clean up text
            content_text = re.sub(r'\s+', ' ', content_text)
            content_text = content_text.strip()
            
            return content_text
        
        except Exception as e:
            logger.debug(f"HTML text extraction failed: {e}")
            return ""
    
    async def search_news(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Search for news articles specifically"""
        # Add news-specific parameters
        news_query = f"{query} news"
        filters = {"time_range": "d"}  # Recent news
        
        return await self.search(news_query, num_results, filters)
    
    async def search_academic(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Search for academic/scholarly content"""
        # Search in academic sites
        academic_query = f"site:scholar.google.com OR site:arxiv.org OR site:pubmed.ncbi.nlm.nih.gov {query}"
        
        return await self.search(academic_query, num_results)
    
    def is_available(self) -> bool:
        """Check if web search service is available"""
        return (
            settings.serp_api_key is not None or 
            settings.google_api_key is not None or
            True  # Fallback is always available
        )
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        self.content_cache.clear() 