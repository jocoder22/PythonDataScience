import os
from bs4 import BeautifulSoup
import requests
import csv

path = 'C:\\Users\\okigboo\\Desktop\\PythonDataScience\\importingData\\webData'
os.chdir(path)
url = 'http://coreyms.com'
response = requests.get(url)
responseText = response.text

soupObject = BeautifulSoup(responseText, features='lxml')

csvfile = open('coreyWeb.csv', 'w')
csv_writer = csv.writer(csvfile)
csv_writer.writerow(['Headline', 'Summary', 'YoutubeLInks'])

for articles in soupObject.find_all('article'):
    headline = articles.h2.a.text
    summary = articles.find('div', class_='entry-content').p.text
    

    try:
        vid_src = articles.find('iframe', class_='youtube-player')['src']

        vid_id = vid_src.split('/')[4]
        vid_id = vid_id.split('?')[0]
   

        youtubelink = f'https://www.youtube.com/watch?=v{vid_id}'
        print(youtubelink)
    except Exception as e:
        youtubelink = None
        

    csv_writer.writerow([headline, summary, youtubelink])
    print()
csvfile.close()
