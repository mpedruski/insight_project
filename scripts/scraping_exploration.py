import requests
import time
from pathlib import Path
import datetime
from bs4 import BeautifulSoup
import re

data_folder = Path("../data/raw")

urls = ['9780061431852','2801102859']
header = 'https://nelligandecouverte.ville.montreal.qc.ca/iii/encore/search/C__S'
seconds = 10
pattern = re.compile(r'[0-9]+ réservation')
pattern1 = re.compile(r'[0-9]+')

digits = []

for item in urls:
    print(datetime.datetime.now())
    url = header + item
    # file_to_open = data_folder / '{}.txt'.format(item)

    page = requests.get(url,time.sleep(seconds))
    soup = BeautifulSoup(page.content,'html.parser')
    a = soup.find_all(class_="holdsMessage")
    try:
        reserv_text = pattern.search(str(a))
        digits.append(int(pattern1.findall(reserv_text.group())[0]))
    except:
        digits.append('NaN')
print(digits)
