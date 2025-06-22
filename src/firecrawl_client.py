"""
Firecrawl client for legal research and web scraping.
Manages API calls to Firecrawl for jurisdictional law research.
"""

import os
import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import time

from dotenv import load_dotenv
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()


@dataclass
class FirecrawlKey:
    """Represents a Firecrawl API key with usage tracking."""
    key: str
    is_active: bool = True
    last_used: float = 0
    error_count: int = 0


class FirecrawlClient:
    """Client for Firecrawl API to perform legal research."""
    
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        self.base_url = "https://api.firecrawl.dev/v0"
        
        # Legal research search engines and domains
        self.legal_sources = {
            "gibraltar": [
                "gibraltarlaws.gov.gi",
                "gibraltarlaw.com",
                "fsc.gi",  # Financial Services Commission
                "gibraltar.gov.gi"
            ],
            "uk": [
                "legislation.gov.uk",
                "gov.uk/inheritance-tax",
                "gov.uk/tax-foreign-income",
                "bailii.org",  # British and Irish Legal Information Institute
                "lawsociety.org.uk"
            ],
            "portugal": [
                "pgdlisboa.pt",
                "dre.pt",  # Diário da República Eletrónico
                "pordata.pt"
            ],
            "international": [
                "oecd.org/tax",
                "europa.eu/taxation",
                "ibanet.org"
            ]
        }
        
        logger.info(f"Initialized Firecrawl client with {len(self.api_keys)} keys")
    
    def _load_api_keys(self) -> List[FirecrawlKey]:
        """Load Firecrawl API keys from environment."""
        keys = []
        
        # Load numbered keys from environment (FIRECRAWL_API_KEY_1, FIRECRAWL_API_KEY_2, etc.)
        key_index = 1
        while True:
            key = os.getenv(f"FIRECRAWL_API_KEY_{key_index}")
            if not key:
                break
            keys.append(FirecrawlKey(key=key.strip()))
            key_index += 1
        
        # Fallback: check for comma-separated FIRECRAWL_KEYS
        if not keys:
            keys_str = os.getenv("FIRECRAWL_KEYS", "")
            if keys_str:
                keys = [FirecrawlKey(key=k.strip()) for k in keys_str.split(",") if k.strip()]
        
        logger.info(f"Loaded {len(keys)} Firecrawl API keys")
        if not keys:
            logger.warning("No Firecrawl API keys found in environment")
        
        return keys
    
    def _get_next_active_key(self) -> Optional[FirecrawlKey]:
        """Get the next active API key."""
        if not self.api_keys:
            return None
            
        attempts = 0
        while attempts < len(self.api_keys):
            key = self.api_keys[self.current_key_index]
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            
            if key.is_active:
                return key
            
            attempts += 1
        
        logger.warning("All Firecrawl keys exhausted")
        return None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=30)
    )
    async def search_legal_info(self, 
                               query: str, 
                               jurisdiction: str,
                               context: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Search for legal information using Firecrawl."""
        api_key = self._get_next_active_key()
        if not api_key:
            logger.error("No active Firecrawl keys available")
            return None
        
        # Build search query with jurisdiction context
        enhanced_query = f"{jurisdiction} law {query}"
        if context:
            enhanced_query += f" {context}"
        
        # Get relevant domains for the jurisdiction
        domains = self.legal_sources.get(jurisdiction.lower(), [])
        
        try:
            async with aiohttp.ClientSession() as session:
                # First, search for relevant pages
                search_results = await self._search_pages(
                    session, api_key.key, enhanced_query, domains
                )
                
                if not search_results:
                    logger.warning(f"No search results for: {enhanced_query}")
                    return None
                
                # Then scrape the most relevant pages
                scraped_content = []
                for result in search_results[:3]:  # Top 3 results
                    url = result.get("url")
                    if url:  # Only scrape if URL exists
                        content = await self._scrape_page(
                            session, api_key.key, url
                        )
                        if content:
                            scraped_content.append({
                                "url": url,
                                "title": result.get("title"),
                                "content": content
                            })
                
                # Update key usage
                api_key.last_used = time.time()
                api_key.error_count = 0
                
                return {
                    "query": query,
                    "jurisdiction": jurisdiction,
                    "search_results": search_results,
                    "scraped_content": scraped_content,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error in legal search: {e}")
            api_key.error_count += 1
            
            if "401" in str(e) or "403" in str(e):
                api_key.is_active = False
                logger.warning(f"Deactivated Firecrawl key due to auth error")
            
            raise
    
    async def _search_pages(self, 
                           session: aiohttp.ClientSession,
                           api_key: str,
                           query: str,
                           domains: List[str]) -> List[Dict]:
        """Search for pages using Firecrawl search endpoint."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Build search with domain restrictions if available
        search_query = query
        if domains:
            # Add site restrictions
            site_queries = " OR ".join([f"site:{domain}" for domain in domains])
            search_query = f"{query} ({site_queries})"
        
        payload = {
            "query": search_query,
            "limit": 10
        }
        
        try:
            async with session.post(
                f"{self.base_url}/search",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("results", [])
                else:
                    error_text = await response.text()
                    logger.error(f"Search failed: {response.status} - {error_text}")
                    return []
                    
        except Exception as e:
            logger.error(f"Search request failed: {e}")
            return []
    
    async def _scrape_page(self,
                          session: aiohttp.ClientSession,
                          api_key: str,
                          url: str) -> Optional[str]:
        """Scrape a specific page using Firecrawl."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "url": url,
            "formats": ["markdown"],  # Get clean markdown
            "onlyMainContent": True,  # Skip navigation, ads, etc.
            "removeImages": True      # Text only for legal content
        }
        
        try:
            async with session.post(
                f"{self.base_url}/scrape",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("markdown", "")
                else:
                    error_text = await response.text()
                    logger.error(f"Scrape failed for {url}: {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Scrape request failed for {url}: {e}")
            return None
    
    async def research_legal_topics(self, topics: List[Dict[str, str]]) -> List[Dict]:
        """Research multiple legal topics."""
        results = []
        
        for topic in topics:
            query = topic.get("query", "")
            jurisdiction = topic.get("jurisdiction", "uk")
            context = topic.get("context", "")
            
            logger.info(f"Researching: {query} in {jurisdiction}")
            
            try:
                result = await self.search_legal_info(query, jurisdiction, context)
                if result:
                    results.append(result)
                
                # Rate limiting between requests
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed to research topic: {query} - {e}")
                continue
        
        return results
    
    def extract_legal_provisions(self, content: str) -> Dict[str, List[str]]:
        """Extract specific legal provisions from scraped content."""
        provisions = {
            "statutes": [],
            "regulations": [],
            "case_law": [],
            "tax_rates": [],
            "deadlines": []
        }
        
        # Extract statute references (e.g., "Section 123 of the...")
        import re
        
        # Statutes and sections
        statute_pattern = r"(?:Section|Article|Regulation)\s+\d+[A-Za-z]?\s+(?:of|in)\s+(?:the\s+)?[\w\s]+"
        provisions["statutes"] = re.findall(statute_pattern, content, re.IGNORECASE)
        
        # Case citations (e.g., "Smith v Jones [2023]")
        case_pattern = r"[\w\s]+\s+v\.?\s+[\w\s]+\s*\[\d{4}\]"
        provisions["case_law"] = re.findall(case_pattern, content)
        
        # Tax rates (e.g., "40% tax", "£325,000 threshold")
        tax_pattern = r"\d+%\s+(?:tax|rate)|£[\d,]+\s+(?:threshold|allowance|exemption)"
        provisions["tax_rates"] = re.findall(tax_pattern, content, re.IGNORECASE)
        
        # Deadlines (e.g., "within 6 months", "12 month period")
        deadline_pattern = r"within\s+\d+\s+(?:days?|months?|years?)|(?:\d+\s+(?:day|month|year))\s+(?:period|deadline|limit)"
        provisions["deadlines"] = re.findall(deadline_pattern, content, re.IGNORECASE)
        
        return provisions
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get current API status."""
        active_keys = sum(1 for k in self.api_keys if k.is_active)
        
        return {
            "total_keys": len(self.api_keys),
            "active_keys": active_keys,
            "key_status": [
                {
                    "key": f"...{k.key[-8:]}",
                    "active": k.is_active,
                    "error_count": k.error_count,
                    "last_used": time.time() - k.last_used if k.last_used > 0 else None
                }
                for k in self.api_keys
            ]
        }


# Example usage
async def test_firecrawl():
    """Test Firecrawl functionality."""
    client = FirecrawlClient()
    
    # Test legal research
    topics = [
        {
            "query": "inheritance tax spouse exemption",
            "jurisdiction": "gibraltar",
            "context": "UK citizen Gibraltar resident"
        },
        {
            "query": "jointly owned property transfer death",
            "jurisdiction": "uk",
            "context": "one owner deceased"
        },
        {
            "query": "domicile determination tax purposes",
            "jurisdiction": "uk",
            "context": "lived abroad Gibraltar"
        }
    ]
    
    results = await client.research_legal_topics(topics)
    
    for result in results:
        print(f"\nResearch Topic: {result['query']}")
        print(f"Jurisdiction: {result['jurisdiction']}")
        print(f"Found {len(result.get('search_results', []))} search results")
        print(f"Scraped {len(result.get('scraped_content', []))} pages")
        
        # Extract provisions from first scraped content
        if result.get('scraped_content'):
            content = result['scraped_content'][0]['content']
            provisions = client.extract_legal_provisions(content)
            
            print("\nExtracted Legal Provisions:")
            for prov_type, items in provisions.items():
                if items:
                    print(f"  {prov_type}: {len(items)} found")
                    for item in items[:2]:  # Show first 2
                        print(f"    - {item}")
    
    # Show API status
    print(f"\nAPI Status: {client.get_api_status()}")


if __name__ == "__main__":
    asyncio.run(test_firecrawl())