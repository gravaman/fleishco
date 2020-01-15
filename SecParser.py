from bs4 import BeautifulSoup
import requests
import sys
import re


class SecParser:
    BASE_URL = 'https://www.sec.gov'
    SEARCH_EDGAR = 'cgi-bin/srch-edgar'
    BROWSE_EDGAR = 'cgi-bin/browse-edgar'

    def __init__(self, verbose=True):
        self.verbose = verbose

    def fetch_filing_links(self, cik, forms=['10-Q'], sy=2018, ey=2020):
        # clean args
        forms = [v.upper() for v in forms]
        sy = max(sy, 1994)
        ey = min(ey, 2020)

        # prep request
        url = '/'.join([self.BASE_URL, self.SEARCH_EDGAR])
        text_param = (
            f'company-cik=({cik}) AND '
            f"type=({'* OR '.join(forms)}) "
        )
        payload = dict(text=text_param, first=sy, last=ey)

        # make req
        try:
            if self.verbose:
                print(f"Fetching {', '.join(forms)} for cik {cik} from {url}")

            res = requests.get(url, params=payload)
            if self.verbose:
                print(f"status code: {res.status_code}")

        except requests.exceptions.RequestException as e:
            print(e)
            sys.exit(1)

        # parse response
        soup = BeautifulSoup(res.text, 'html.parser')
        table = soup.find_all('table')[4]
        regx = re.compile(r'index\.htm')
        links = [row.find('a', href=regx) for row in table.find_all('tr')]
        return [link['href'] for link in links if link]

    def fetch_xbrli_link(self, href):
        # clean args and prep request
        if not href.startswith('/'):
            href = '/'+href

        url = self.BASE_URL+href

        # make req
        try:
            if self.verbose:
                print(f'Fetching xbrli for {href}')

            res = requests.get(url)
            if self.verbose:
                print(f'status code: {res.status_code}')

        except requests.exceptions.RequestException as e:
            print(e)
            sys.exit(1)

        # parse res
        soup = BeautifulSoup(res.text, 'html.parser')
        table = soup.find('table', class_='tableFile', summary='Data Files')
        href = None
        link = table.find('a', href=re.compile(r'_htm.xml'))
        if link:
            href = link['href']
        return href


if __name__ == '__main__':
    sixcik = '0000701374'
    forms = ['10-q', '10-k']
    sy = 2018
    ey = 2021

    secp = SecParser()

    if False:
        links = secp.fetch_filing_links(sixcik, forms=forms, sy=sy, ey=ey)
        print(f'\nFiling Links:')
        for link in links:
            print(link)

    href = (
            f'/Archives/edgar/data/701374/000070137419000147'
            f'/0000701374-19-000147-index.htm'
    )
    if True:
        link = secp.fetch_xbrli_link(href)
        print(f'link: {link}')
