import os
import sys
from pathlib import Path
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
import re
import json
from urllib.parse import urljoin
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, LEAN4_DOCS_URL
from src.data.loader import LeanLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def discover_urls(base_url: str) -> List[str]:
    """
    Discover documentation URLs by crawling from the base URL.
    
    Uses a simple crawler based on requests and BeautifulSoup.
    
    Args:
        base_url: Base URL for the Lean 4 documentation
        
    Returns:
        List of URLs to scrape
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin, urlparse
        
        logger.info(f"Discovering URLs from {base_url}")
        
        # Function to normalize URLs
        def normalize_url(url):
            parsed = urlparse(url)
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        
        # Get base domain
        base_domain = urlparse(base_url).netloc
        
        # Set to store discovered URLs
        discovered_urls = set([base_url])
        to_visit = [base_url]
        visited = set()
        
        # Simple crawler
        max_urls = 100  # Limit to prevent excessive crawling
        while to_visit and len(discovered_urls) < max_urls:
            current_url = to_visit.pop(0)
            if current_url in visited:
                continue
                
            visited.add(current_url)
            
            try:
                response = requests.get(current_url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    
                    # Find all links
                    for link in soup.find_all("a", href=True):
                        href = link["href"]
                        full_url = urljoin(current_url, href)
                        normalized = normalize_url(full_url)
                        
                        # Only process URLs from the same domain and containing 'doc'
                        if urlparse(full_url).netloc == base_domain and "/doc/" in full_url:
                            if normalized not in discovered_urls and normalized not in visited:
                                discovered_urls.add(normalized)
                                to_visit.append(normalized)
            except Exception as e:
                logger.warning(f"Error processing {current_url}: {e}")
                
        urls = list(discovered_urls)
        logger.info(f"Discovered {len(urls)} URLs")
        return urls
        
    except ImportError as e:
        logger.warning(f"Error with URL discovery: {e}. Falling back to manual URL list.")
        
        # If CrawlAI is not available, use a list of common documentation paths
        base_paths = [
            "",  # Root
            "api/",
            "examples/",
            "tutorial/",
            "reference/",
            "guide/",
        ]
        
        urls = [urljoin(base_url, path) for path in base_paths]
        logger.info(f"Using manual list of {len(urls)} URLs")
        
        return urls
    
    except Exception as e:
        logger.error(f"Error during URL discovery: {e}")
        return [base_url]  # Fall back to just the base URL

def scrape_documentation(urls: List[str], output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Scrape Lean 4 documentation from the provided URLs.
    
    Args:
        urls: List of URLs to scrape
        output_dir: Directory to save scraped content
        
    Returns:
        List of scraped documents
    """
    if output_dir is None:
        output_dir = RAW_DATA_DIR / "docs"
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Scraping documentation from {len(urls)} URLs...")
    
    # Load documentation
    loader = LeanLoader()
    documents = loader.load_lean_docs(urls)
    
    # Save raw content
    for i, doc in enumerate(documents):
        file_path = os.path.join(output_dir, f"doc_{i:04d}.json")
        with open(file_path, "w") as f:
            json.dump({
                "url": doc.metadata.get("source", "unknown"),
                "title": doc.metadata.get("title", "Untitled"),
                "content": doc.page_content
            }, f, indent=2)
    
    logger.info(f"Saved {len(documents)} documents to {output_dir}")
    
    return documents

def main():
    parser = argparse.ArgumentParser(description="Scrape Lean 4 documentation")
    parser.add_argument("--base-url", default=LEAN4_DOCS_URL, help="Base URL for Lean 4 documentation")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--urls", nargs="+", help="Specific URLs to scrape")
    
    args = parser.parse_args()
    
    if args.urls:
        urls = args.urls
    else:
        urls = discover_urls(args.base_url)
    
    scrape_documentation(urls, args.output)

if __name__ == "__main__":
    main()