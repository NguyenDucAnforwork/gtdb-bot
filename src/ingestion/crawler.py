# src/ingestion/crawler.py
"""
Web crawler cho thuvienphapluat.vn
Crawl văn bản pháp luật từ URL và trích xuất nội dung
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional
import re

class ThuvienPhaplaatCrawler:
    """Crawler cho thuvienphapluat.vn"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def crawl(self, url: str) -> Optional[Dict[str, str]]:
        """
        Crawl văn bản từ URL
        
        Args:
            url: URL văn bản trên thuvienphapluat.vn
            
        Returns:
            Dict với 'law_code', 'title', 'content' hoặc None nếu lỗi
        """
        try:
            print(f"🌐 Crawling: {url}")
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract law code và title
            law_code = self._extract_law_code(soup, url)
            title = self._extract_title(soup)
            content = self._extract_content(soup)
            
            if not content:
                print("❌ Không tìm thấy nội dung văn bản")
                return None
            
            print(f"✅ Crawled: {law_code} - {title}")
            print(f"   Content length: {len(content)} chars")
            
            return {
                'law_code': law_code,
                'title': title,
                'content': content,
                'url': url
            }
            
        except Exception as e:
            print(f"❌ Crawl failed: {e}")
            return None
    
    def _extract_law_code(self, soup: BeautifulSoup, url: str) -> str:
        """Trích xuất mã văn bản (VD: 158/2024/NĐ-CP)"""
        
        # Method 1: Từ URL pattern
        # https://thuvienphapluat.vn/van-ban/.../Nghi-dinh-158-2024-ND-CP-...
        url_match = re.search(r'(Nghi-dinh|Luat|Thong-tu|Quyet-dinh)-(\d+-\d+-[A-Z-]+)', url)
        if url_match:
            doc_type = url_match.group(1).replace('-', ' ')
            code = url_match.group(2).replace('-', '/')
            
            # Map document type
            type_map = {
                'Nghi dinh': 'NĐ-CP',
                'Luat': 'QH',
                'Thong tu': 'TT',
                'Quyet dinh': 'QĐ'
            }
            
            # Clean code (158/2024/ND/CP -> 158/2024/NĐ-CP)
            for vn, abbr in type_map.items():
                if code.endswith(abbr.replace('-', '/')):
                    code = code.replace(abbr.replace('-', '/'), abbr)
                    break
            
            return code
        
        # Method 2: Từ breadcrumb hoặc title
        breadcrumb = soup.find('ol', class_='breadcrumb')
        if breadcrumb:
            links = breadcrumb.find_all('a')
            for link in links:
                text = link.get_text(strip=True)
                code_match = re.search(r'(\d+/\d+/[A-Z-]+)', text)
                if code_match:
                    return code_match.group(1)
        
        # Fallback: "unknown"
        return "unknown"
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Trích xuất tiêu đề văn bản"""
        
        # Method 1: h1.document-title
        title_elem = soup.find('h1', class_='document-title')
        if title_elem:
            return title_elem.get_text(strip=True)
        
        # Method 2: .title-document
        title_elem = soup.find('div', class_='title-document')
        if title_elem:
            return title_elem.get_text(strip=True)
        
        # Method 3: <title> tag
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True)
        
        return "Không có tiêu đề"
    
    def _extract_content(self, soup: BeautifulSoup) -> Optional[str]:
        """Trích xuất nội dung văn bản"""
        
        # Method 1: div#content1 (thuvienphapluat.vn structure)
        content_div = soup.find('div', id='content1')
        if content_div:
            return self._clean_content(content_div.get_text())
        
        # Method 2: .document-content
        content_div = soup.find('div', class_='document-content')
        if content_div:
            return self._clean_content(content_div.get_text())
        
        # Method 3: article tag
        article = soup.find('article')
        if article:
            return self._clean_content(article.get_text())
        
        return None
    
    def _clean_content(self, text: str) -> str:
        """Làm sạch nội dung (remove extra whitespace, etc.)"""
        
        # Remove multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text


def crawl_document(url: str) -> Optional[Dict[str, str]]:
    """
    Convenience function để crawl document
    
    Args:
        url: URL của văn bản
        
    Returns:
        Document dict hoặc None
    """
    crawler = ThuvienPhaplaatCrawler()
    return crawler.crawl(url)
