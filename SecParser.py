from bs4 import BeautifulSoup
import requests
import sys
import re


class Gaap:
    DEI_NAMES = [
        'documenttype', 'entitycentralindexkey',
        'currentfiscalyearenddate',
        'documentfiscalyearfocus', 'documentfiscalperiodfocus',
        'documentperiodenddate', 'entityregistrantname', 'tradingsymbol',
        'securityexchangename', 'entitycommonstocksharesoutstanding'
    ]

    DEI_TO_REC = {
        'documenttype': 'doctype', 'entitycentralindexkey': 'cik',
        'currentfiscalyearenddate': 'fye', 'documentfiscalyearfocus': 'yrfoc',
        'documentfiscalperiodfocus': 'pfoc', 'documentperiodenddate': 'pend',
        'entityregistrantname': 'entity', 'tradingsymbol': 'sym',
        'securityexchangename': 'exch',
        'entitycommonstocksharesoutstanding': 'sharesos'
    }

    def __init__(self, xbrl):
        """
            dei tags:
            documenttype: 10-Q
            entitycentralindexkey: 00007013741900047
            currentfiscalyearenddate: --12-31
            documentfiscalyearfocus: 2019
            documentfiscalperiodfocus: Q3
            documentperiodenddate: 09-30
            entityregistrantname: Six Flags Entertainment Corporation
            tradingsymbol: SIX
            securityexchangename: NYSE
            entitycommonstocksharesoutstanding: 84524441 (most recent)
        """
        self.xbrl = xbrl

        # get dei data
        dts = [xbrl.find('dei:'+name) for name in self.DEI_NAMES]
        dts = [tag for tag in dts if tag]
        self.dei_tags = {}
        for tag in dts:
            name = tag.name[4:]
            if name in self.DEI_TO_REC:
                key = self.DEI_TO_REC[name]
                self.dei_tags[key] = tag.text

        # get gaap data
        nameregx = re.compile(r'^us-gaap:')
        numregx = re.compile(r'^\d+(\.\d+)?$')
        tags = xbrl.find_all(name=nameregx, string=numregx)
        self.records = []
        for tag in tags:
            attrs = {
                'name': tag.name.split(':')[-1],
                'amnt': tag.text
            }
            attrs = {**attrs, **self.dei_tags}
            period = self.get_period(tag)
            attrs = {**attrs, **period}
            rec = GaapRecord(**attrs)
            self.records.append(rec)

    def get_period(self, tag):
        """
            context period tag will either be name instant or have start/end
            based on whether context id has Duration or As_Of
        """
        context = self.xbrl.find(name='context', id=tag['contextref'])
        ptags = context.period.find_all()

        # check if instant or duration
        period = {}
        for ptag in ptags:
            if ptag.name == 'instant' or ptag.name == 'enddate':
                period['ed'] = ptag.text
            elif ptag.name == 'startdate':
                period['sd'] = ptag.text
            else:
                print(f'Unknown period tag name: {ptag.name}')

        return period


class GaapRecord:
    REQ_KEYS = ['cik', 'name', 'doctype', 'fye', 'yrfoc',
                'pfoc', 'pend', 'amnt']
    OPT_KEYS = ['sym', 'exch', 'sd', 'entity']

    def __init__(self, **kwargs):
        """
            cik: 0000701374190047
            name: commonstocksharesissued
            doctype: 10-Q
            fye: --12-31
            yrfoc: 2019
            pfoc: Q3
            pend: 09-30
            amnt: 100032123

            kwargs:
            sym: SIX
            exch: NYSE
            sd: 2019-09-30
            ed: 2019-09-30
            entity: Six Flags Entertainment Corporation
        """
        for k in self.REQ_KEYS:
            if k not in kwargs.keys():
                raise ValueError(f'GaapRecord key miss: {k}')

        all_keys = self.REQ_KEYS+self.OPT_KEYS
        self.attr_keys = []
        for k, v in kwargs.items():
            if k in all_keys:
                self.attr_keys.append(k)
                setattr(self, k, v)

    def __str__(self):
        outs = [f'{k}: {getattr(self, k)}' for k in self.attr_keys]
        return '\n'.join(outs)


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
        res = self.fetch(url, params=payload)
        soup = BeautifulSoup(res.text, 'html.parser')
        table = soup.find_all('table')[4]
        regx = re.compile(r'index\.htm')
        links = [row.find('a', href=regx) for row in table.find_all('tr')]
        return [link['href'] for link in links if link]

    def fetch_xbrli_link(self, href):
        url = self.url_from_href(href)
        res = self.fetch(url)
        soup = BeautifulSoup(res.text, 'html.parser')
        table = soup.find('table', class_='tableFile', summary='Data Files')
        href = None
        link = table.find('a', href=re.compile(r'_htm.xml'))
        if link:
            href = link['href']
        return href

    def fetch_xbrl_data(self, href):
        # clean args and prep request
        url = self.url_from_href(href)
        res = self.fetch(url)
        soup = BeautifulSoup(res.text, 'lxml')

        xbrl = soup.find('xbrl')
        return Gaap(xbrl)

    def fetch(self, url, params=None):
        try:
            if self.verbose:
                print(f'fetching {href}')

            res = requests.get(url, params=params)
            if self.verbose:
                print(f'fetch status code: {res.status_code}')

        except requests.exceptions.RequestException as e:
            print(e)
            sys.exit(1)

        return res

    def url_from_href(self, href):
        if not href.startswith('/'):
            href = '/'+href

        return self.BASE_URL+href


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

    if False:
        href = (
            f'/Archives/edgar/data/701374/000070137419000147'
            f'/0000701374-19-000147-index.htm'
        )
        link = secp.fetch_xbrli_link(href)
        print(f'link: {link}')

    if False:
        href = (
            f'/Archives/edgar/data/701374/000070137419000147'
            f'/six-20190930x10q_htm.xml'
        )
        gaap = secp.fetch_xbrl_data(href)
        for rec in gaap.records[:10]:
            print(rec)
