import requests, redis, re
from bs4 import BeautifulSoup
from datetime import datetime

def fetch_holiday():
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    url = "https://www.myhora.com/calendar/ical/holiday.aspx?latest.txt"
    thai_months = {'ม.ค.': '01', 'ก.พ.': '02', 'มี.ค.': '03', 'เม.ย.': '04',
                   'พ.ค.': '05', 'มิ.ย.': '06', 'ก.ค.': '07', 'ส.ค.': '08',
                   'ก.ย.': '09', 'ต.ค.': '10', 'พ.ย.': '11', 'ธ.ค.': '12'}

    def convert(date_str):
        match = re.match(r'(\d{1,2})\s(\S+)\s(\d{4})', date_str.strip())
        if not match: return None
        d, m, y = match.groups()
        return f"{int(d):02d}/{thai_months.get(m)}/{int(y) - 543}"

    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')

    for div in soup.find_all("div", class_="mb-5"):
        cols = div.find_all("div")
        if len(cols) >= 2:
            raw = cols[0].text.strip()
            name = cols[1].text.strip()
            formatted = convert(raw)
            if formatted:
                iso = datetime.strptime(formatted, "%d/%m/%Y").strftime("%Y-%m-%d")
                r.set(f"holiday:{iso}", name)
                r.sadd(f"holidays:{iso[:4]}", iso)
