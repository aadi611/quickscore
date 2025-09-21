"""
Intelligent web scraper service for LinkedIn, websites, and GitHub data extraction.
"""
import logging
import asyncio
import random
import re
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urljoin, urlparse
import json

import httpx
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Browser, Page, TimeoutError as PlaywrightTimeoutError

from app.core.config import settings

logger = logging.getLogger(__name__)


class IntelligentScraper:
    """Intelligent web scraper with anti-detection capabilities."""
    
    def __init__(self):
        self.supported_sources = ["linkedin", "website", "github"]
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        ]
        self.github_token = settings.GITHUB_TOKEN if hasattr(settings, 'GITHUB_TOKEN') else None
        
    async def scrape_linkedin_profile(self, url: str) -> Dict[str, Any]:
        """
        Scrape LinkedIn profile with anti-detection measures.
        
        Note: This implementation is for educational purposes. 
        In production, use LinkedIn's official API when possible.
        """
        logger.info(f"Scraping LinkedIn profile: {url}")
        
        try:
            async with async_playwright() as p:
                # Launch browser with stealth mode
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-blink-features=AutomationControlled',
                        '--disable-dev-shm-usage',
                        '--disable-extensions',
                        '--disable-gpu',
                        '--disable-web-security',
                        '--no-first-run',
                        '--no-default-browser-check'
                    ]
                )
                
                # Create context with random user agent
                context = await browser.new_context(
                    user_agent=random.choice(self.user_agents),
                    viewport={'width': 1920, 'height': 1080},
                    java_script_enabled=True
                )
                
                page = await context.new_page()
                
                # Set additional headers to appear more human-like
                await page.set_extra_http_headers({
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                })
                
                # Navigate to LinkedIn profile
                await page.goto(url, wait_until='networkidle', timeout=30000)
                
                # Random delay to appear more human-like
                await asyncio.sleep(random.uniform(2, 5))
                
                # Extract profile data
                profile_data = await self._extract_linkedin_data(page)
                
                await browser.close()
                
                return {
                    "success": True,
                    "url": url,
                    "data": profile_data,
                    "timestamp": "2024-01-01T00:00:00Z"
                }
                
        except Exception as e:
            logger.error(f"LinkedIn scraping failed for {url}: {e}")
            return {
                "success": False,
                "url": url,
                "error": str(e),
                "data": {}
            }
    
    async def _extract_linkedin_data(self, page: Page) -> Dict[str, Any]:
        """Extract structured data from LinkedIn profile page."""
        data = {
            "name": "",
            "headline": "",
            "location": "",
            "connections": "",
            "work_experience": [],
            "education": [],
            "skills": [],
            "recommendations_count": 0,
            "posts_engagement": {}
        }
        
        try:
            # Extract name
            try:
                name_element = await page.wait_for_selector('h1.text-heading-xlarge', timeout=10000)
                if name_element:
                    data["name"] = await name_element.inner_text()
            except:
                logger.warning("Could not extract name from LinkedIn profile")
            
            # Extract headline
            try:
                headline_element = await page.query_selector('.text-body-medium.break-words')
                if headline_element:
                    data["headline"] = await headline_element.inner_text()
            except:
                logger.warning("Could not extract headline from LinkedIn profile")
            
            # Extract location
            try:
                location_element = await page.query_selector('.text-body-small.inline.t-black--light.break-words')
                if location_element:
                    data["location"] = await location_element.inner_text()
            except:
                logger.warning("Could not extract location from LinkedIn profile")
            
            # Extract connections count
            try:
                connections_element = await page.query_selector('span.t-black--light.t-normal')
                if connections_element:
                    connections_text = await connections_element.inner_text()
                    # Extract number from text like "500+ connections"
                    connections_match = re.search(r'(\d+)', connections_text)
                    if connections_match:
                        data["connections"] = connections_match.group(1)
            except:
                logger.warning("Could not extract connections from LinkedIn profile")
            
            # Extract work experience
            try:
                experience_section = await page.query_selector('#experience')
                if experience_section:
                    experience_items = await page.query_selector_all('.pvs-list__paged-list-item')
                    
                    for item in experience_items[:5]:  # Limit to 5 most recent
                        try:
                            title_elem = await item.query_selector('.mr1.t-bold')
                            company_elem = await item.query_selector('.t-14.t-normal')
                            duration_elem = await item.query_selector('.t-14.t-normal.t-black--light')
                            
                            experience = {
                                "title": await title_elem.inner_text() if title_elem else "",
                                "company": await company_elem.inner_text() if company_elem else "",
                                "duration": await duration_elem.inner_text() if duration_elem else ""
                            }
                            
                            if experience["title"] or experience["company"]:
                                data["work_experience"].append(experience)
                        except:
                            continue
            except:
                logger.warning("Could not extract work experience from LinkedIn profile")
            
            # Extract education
            try:
                education_section = await page.query_selector('#education')
                if education_section:
                    education_items = await page.query_selector_all('.pvs-list__paged-list-item')
                    
                    for item in education_items[:3]:  # Limit to 3 most recent
                        try:
                            school_elem = await item.query_selector('.mr1.hoverable-link-text.t-bold')
                            degree_elem = await item.query_selector('.t-14.t-normal')
                            
                            education = {
                                "school": await school_elem.inner_text() if school_elem else "",
                                "degree": await degree_elem.inner_text() if degree_elem else ""
                            }
                            
                            if education["school"] or education["degree"]:
                                data["education"].append(education)
                        except:
                            continue
            except:
                logger.warning("Could not extract education from LinkedIn profile")
            
        except Exception as e:
            logger.error(f"Error extracting LinkedIn data: {e}")
        
        return data
    
    async def scrape_company_website(self, url: str) -> Dict[str, Any]:
        """Scrape company website for team, product, and business information."""
        logger.info(f"Scraping company website: {url}")
        
        try:
            async with httpx.AsyncClient(
                timeout=30.0,
                headers={
                    'User-Agent': random.choice(self.user_agents),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive'
                }
            ) as client:
                
                # Get main page
                response = await client.get(url)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract structured data
                website_data = {
                    "team_members": await self._extract_team_members(soup, client, url),
                    "product_features": await self._extract_product_features(soup),
                    "pricing_info": await self._extract_pricing_info(soup),
                    "customer_logos": await self._extract_customer_logos(soup),
                    "tech_stack": await self._extract_tech_stack(soup),
                    "company_info": await self._extract_company_info(soup),
                    "contact_info": await self._extract_contact_info(soup)
                }
                
                return {
                    "success": True,
                    "url": url,
                    "data": website_data,
                    "timestamp": "2024-01-01T00:00:00Z"
                }
                
        except Exception as e:
            logger.error(f"Website scraping failed for {url}: {e}")
            return {
                "success": False,
                "url": url,
                "error": str(e),
                "data": {}
            }
    
    async def _extract_team_members(self, soup: BeautifulSoup, client: httpx.AsyncClient, base_url: str) -> List[Dict[str, str]]:
        """Extract team member information from website."""
        team_members = []
        
        # Common team page patterns
        team_links = []
        
        # Look for team/about links
        for link in soup.find_all('a', href=True):
            href = link['href'].lower()
            text = link.get_text().lower()
            
            if any(keyword in href or keyword in text for keyword in ['team', 'about', 'founders', 'leadership']):
                full_url = urljoin(base_url, link['href'])
                team_links.append(full_url)
        
        # Try to scrape team pages
        for team_url in team_links[:3]:  # Limit to 3 team pages
            try:
                response = await client.get(team_url)
                team_soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for team member cards/profiles
                for element in team_soup.find_all(['div', 'section'], class_=re.compile(r'team|member|founder|staff', re.I)):
                    name_elem = element.find(['h1', 'h2', 'h3', 'h4'], string=re.compile(r'^[A-Z][a-z]+ [A-Z][a-z]+'))
                    title_elem = element.find(['p', 'span', 'div'], string=re.compile(r'(CEO|CTO|Founder|VP|Director|Manager)', re.I))
                    
                    if name_elem:
                        member = {
                            "name": name_elem.get_text().strip(),
                            "title": title_elem.get_text().strip() if title_elem else "",
                            "source_url": team_url
                        }
                        team_members.append(member)
                
                if len(team_members) >= 10:  # Limit results
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to scrape team page {team_url}: {e}")
                continue
        
        return team_members[:10]  # Return max 10 team members
    
    async def _extract_product_features(self, soup: BeautifulSoup) -> List[str]:
        """Extract product features from website."""
        features = []
        
        # Look for features sections
        feature_sections = soup.find_all(['section', 'div'], class_=re.compile(r'feature|product|benefit', re.I))
        
        for section in feature_sections:
            # Look for feature lists
            feature_lists = section.find_all(['ul', 'ol'])
            for feature_list in feature_lists:
                items = feature_list.find_all('li')
                for item in items:
                    text = item.get_text().strip()
                    if len(text) > 10 and len(text) < 200:  # Filter reasonable length features
                        features.append(text)
        
        # Also look for headings that might be features
        for heading in soup.find_all(['h2', 'h3', 'h4']):
            text = heading.get_text().strip()
            if len(text) > 5 and len(text) < 100:
                # Check if it looks like a feature (not navigation)
                if not any(nav_word in text.lower() for nav_word in ['home', 'about', 'contact', 'blog', 'news']):
                    features.append(text)
        
        return list(set(features[:20]))  # Return unique features, max 20
    
    async def _extract_pricing_info(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract pricing information from website."""
        pricing_info = {
            "plans": [],
            "pricing_model": "",
            "free_tier": False
        }
        
        # Look for pricing sections
        pricing_sections = soup.find_all(['section', 'div'], class_=re.compile(r'pricing|plan|price', re.I))
        
        for section in pricing_sections:
            # Look for price amounts
            price_elements = section.find_all(string=re.compile(r'\$\d+'))
            for price in price_elements:
                pricing_info["plans"].append(price.strip())
            
            # Check for free tier indicators
            free_indicators = section.find_all(string=re.compile(r'free|trial|demo', re.I))
            if free_indicators:
                pricing_info["free_tier"] = True
        
        # Determine pricing model
        text_content = soup.get_text().lower()
        if 'subscription' in text_content or 'monthly' in text_content:
            pricing_info["pricing_model"] = "subscription"
        elif 'one-time' in text_content or 'lifetime' in text_content:
            pricing_info["pricing_model"] = "one-time"
        elif 'usage' in text_content or 'pay-as-you' in text_content:
            pricing_info["pricing_model"] = "usage-based"
        
        return pricing_info
    
    async def _extract_customer_logos(self, soup: BeautifulSoup) -> List[str]:
        """Extract customer/client logos and names."""
        customers = []
        
        # Look for customer/client sections
        customer_sections = soup.find_all(['section', 'div'], class_=re.compile(r'customer|client|partner|testimonial', re.I))
        
        for section in customer_sections:
            # Look for images that might be logos
            images = section.find_all('img')
            for img in images:
                alt_text = img.get('alt', '')
                src = img.get('src', '')
                
                if alt_text and len(alt_text) < 50:  # Reasonable logo alt text
                    customers.append(alt_text)
                elif 'logo' in src.lower():
                    # Extract company name from logo filename
                    filename = src.split('/')[-1].split('.')[0]
                    customers.append(filename.replace('-', ' ').replace('_', ' ').title())
        
        return list(set(customers[:15]))  # Return unique customers, max 15
    
    async def _extract_tech_stack(self, soup: BeautifulSoup) -> List[str]:
        """Extract technology stack information."""
        tech_stack = []
        
        # Look for technology mentions in text
        text_content = soup.get_text().lower()
        
        # Common technologies to look for
        technologies = [
            'react', 'vue', 'angular', 'python', 'javascript', 'typescript', 'java', 'go', 'rust',
            'node.js', 'express', 'django', 'flask', 'rails', 'spring', 'laravel',
            'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform',
            'api', 'rest', 'graphql', 'microservices', 'serverless'
        ]
        
        for tech in technologies:
            if tech in text_content:
                tech_stack.append(tech.title())
        
        # Also look for tech stack sections specifically
        tech_sections = soup.find_all(['section', 'div'], class_=re.compile(r'tech|stack|technology', re.I))
        for section in tech_sections:
            section_text = section.get_text().lower()
            for tech in technologies:
                if tech in section_text and tech.title() not in tech_stack:
                    tech_stack.append(tech.title())
        
        return tech_stack[:10]  # Return max 10 technologies
    
    async def _extract_company_info(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract general company information."""
        info = {
            "description": "",
            "industry": "",
            "founded": "",
            "location": ""
        }
        
        # Look for meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            info["description"] = meta_desc.get('content', '')
        
        # Look for about/company info
        about_sections = soup.find_all(['section', 'div'], class_=re.compile(r'about|company|description', re.I))
        for section in about_sections:
            text = section.get_text().strip()
            if len(text) > 50 and len(text) < 500:  # Reasonable description length
                info["description"] = text
                break
        
        # Look for founded year
        text_content = soup.get_text()
        founded_match = re.search(r'founded.{0,20}(\d{4})', text_content, re.I)
        if founded_match:
            info["founded"] = founded_match.group(1)
        
        return info
    
    async def _extract_contact_info(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract contact information."""
        contact = {
            "email": "",
            "phone": "",
            "address": ""
        }
        
        text_content = soup.get_text()
        
        # Look for email
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text_content)
        if email_match:
            contact["email"] = email_match.group()
        
        # Look for phone
        phone_match = re.search(r'\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}', text_content)
        if phone_match:
            contact["phone"] = phone_match.group()
        
        return contact
    
    async def extract_github_metrics(self, github_url: str) -> Dict[str, Any]:
        """Extract GitHub repository metrics using GitHub API."""
        logger.info(f"Extracting GitHub metrics for: {github_url}")
        
        try:
            # Parse GitHub URL to get owner and repo
            path_parts = urlparse(github_url).path.strip('/').split('/')
            if len(path_parts) < 2:
                raise ValueError("Invalid GitHub URL format")
            
            owner, repo = path_parts[0], path_parts[1]
            
            headers = {
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'QuickScore-Analyzer'
            }
            
            if self.github_token:
                headers['Authorization'] = f'token {self.github_token}'
            
            async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
                # Get repository information
                repo_response = await client.get(f'https://api.github.com/repos/{owner}/{repo}')
                repo_response.raise_for_status()
                repo_data = repo_response.json()
                
                # Get contributor statistics
                contributors_response = await client.get(f'https://api.github.com/repos/{owner}/{repo}/contributors')
                contributors_data = contributors_response.json() if contributors_response.status_code == 200 else []
                
                # Get languages
                languages_response = await client.get(f'https://api.github.com/repos/{owner}/{repo}/languages')
                languages_data = languages_response.json() if languages_response.status_code == 200 else {}
                
                # Get recent commits (for activity analysis)
                commits_response = await client.get(f'https://api.github.com/repos/{owner}/{repo}/commits?per_page=100')
                commits_data = commits_response.json() if commits_response.status_code == 200 else []
                
                # Process and calculate metrics
                metrics = {
                    "repo_stats": {
                        "stars": repo_data.get('stargazers_count', 0),
                        "forks": repo_data.get('forks_count', 0),
                        "watchers": repo_data.get('watchers_count', 0),
                        "open_issues": repo_data.get('open_issues_count', 0),
                        "size_kb": repo_data.get('size', 0),
                        "created_at": repo_data.get('created_at'),
                        "updated_at": repo_data.get('updated_at'),
                        "pushed_at": repo_data.get('pushed_at')
                    },
                    "contributors": {
                        "total_contributors": len(contributors_data),
                        "top_contributors": [
                            {
                                "login": contrib.get('login'),
                                "contributions": contrib.get('contributions', 0)
                            }
                            for contrib in contributors_data[:5]
                        ]
                    },
                    "languages": {
                        "primary_language": repo_data.get('language'),
                        "all_languages": list(languages_data.keys()),
                        "language_bytes": languages_data
                    },
                    "activity": {
                        "recent_commits_count": len(commits_data),
                        "commit_frequency": self._calculate_commit_frequency(commits_data),
                        "last_commit_date": commits_data[0].get('commit', {}).get('author', {}).get('date') if commits_data else None
                    },
                    "health_score": self._calculate_repo_health_score(repo_data, contributors_data, commits_data)
                }
                
                return {
                    "success": True,
                    "url": github_url,
                    "data": metrics,
                    "timestamp": "2024-01-01T00:00:00Z"
                }
                
        except Exception as e:
            logger.error(f"GitHub metrics extraction failed for {github_url}: {e}")
            return {
                "success": False,
                "url": github_url,
                "error": str(e),
                "data": {}
            }
    
    def _calculate_commit_frequency(self, commits_data: List[Dict]) -> Dict[str, float]:
        """Calculate commit frequency metrics."""
        if not commits_data:
            return {"commits_per_week": 0.0, "active_days": 0}
        
        # Simple calculation - in production would be more sophisticated
        total_commits = len(commits_data)
        
        # Estimate based on recent 100 commits
        commits_per_week = total_commits / 4.0  # Rough estimate
        
        # Count unique commit dates
        unique_dates = set()
        for commit in commits_data:
            date = commit.get('commit', {}).get('author', {}).get('date', '')
            if date:
                unique_dates.add(date.split('T')[0])  # Extract date part
        
        return {
            "commits_per_week": commits_per_week,
            "active_days": len(unique_dates)
        }
    
    def _calculate_repo_health_score(self, repo_data: Dict, contributors_data: List, commits_data: List) -> float:
        """Calculate a health score for the repository (0-100)."""
        score = 0.0
        
        # Stars factor (0-25 points)
        stars = repo_data.get('stargazers_count', 0)
        if stars > 1000:
            score += 25
        elif stars > 100:
            score += 20
        elif stars > 10:
            score += 15
        elif stars > 0:
            score += 10
        
        # Contributors factor (0-25 points)
        contributor_count = len(contributors_data)
        if contributor_count > 10:
            score += 25
        elif contributor_count > 5:
            score += 20
        elif contributor_count > 2:
            score += 15
        elif contributor_count > 0:
            score += 10
        
        # Recent activity factor (0-25 points)
        recent_commits = len(commits_data)
        if recent_commits > 50:
            score += 25
        elif recent_commits > 20:
            score += 20
        elif recent_commits > 10:
            score += 15
        elif recent_commits > 0:
            score += 10
        
        # Repository maintenance factor (0-25 points)
        if repo_data.get('has_readme'):
            score += 10
        if repo_data.get('has_license'):
            score += 10
        if repo_data.get('open_issues_count', 0) < 20:
            score += 5
        
        return min(100.0, score)