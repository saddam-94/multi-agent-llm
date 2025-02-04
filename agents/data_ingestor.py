import requests
from bs4 import BeautifulSoup


class Data_Ingestor:
    def __init__(self, user_agent=None):
        self.user_agent = user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        self.headers = {"User-Agent": self.user_agent}

    def google_search_scrape(self, query, websites, num_results=10):
        search_results = []
        for website in websites:
            search_query = f"{query} {website}"
            search_url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}&num={num_results}"
            response = requests.get(search_url, headers=self.headers)

            if response.status_code != 200:
                print(f"Error fetching search results for {search_query}: {response.status_code}")
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            for g in soup.find_all('div', class_='tF2Cxc'):
                title = g.find('h3').text if g.find('h3') else "No Title"
                link = g.find('a')['href'] if g.find('a') else "No Link"
                search_results.append({"website": website, "title": title, "link": link})

        return search_results

    def scrape_page_content(self, url):
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            return " ".join([p.text for p in paragraphs])
        else:
            print(f"Failed to fetch {url}")
            return ""

    def get_content_from_query(self, query, websites):
        search_results = self.google_search_scrape(query, websites, num_results=5)

        final_content = ""
        for result in search_results:
            content = self.scrape_page_content(result["link"])
            if len(content) < 20:
              continue

            final_content += content + "\n"

        return final_content
