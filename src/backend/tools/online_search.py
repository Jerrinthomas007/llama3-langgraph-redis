import requests
from bs4 import BeautifulSoup
import trafilatura


def duckduckgo_search(query):
    headers={
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; win64; x64) ApplewebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
      
    }
    url = f'https://html.duckduckgo.com/html/?q={query}'
    response = requests.get(url,headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    results =[]
    for i, result in enumerate(soup.find_all('div', class_='result'),start=1):
        if i>10:
            break

        title_tag =  result.find('a', class_='result__a')
        if not title_tag:
            continue

        link = title_tag['href']
        snippet_tag= result.find('a',class_='result__snippet')
        snippet =snippet_tag.text.strip() if snippet_tag else 'NO description available'

        results.append({
            'id':i,
            'link': link,
            'search_description':snippet
        })

    return results

def scrape_webpage(url):
    try:
        downloaded = trafilatura.fetch_url(url=url)
        return trafilatura.extract(downloaded, include_formatting=True, include_links=True)
    except Exception as e:
        return None
    
