#!/usr/bin/env python

import requests
from bs4 import BeautifulSoup

# Specify the url: url
url = "http://www.datacamp.com/teach/documentation"
url2 = 'https://www.crummy.com/software/BeautifulSoup/'

# Packages the request, send the request and catch the response: r
r = requests.get(url)

# Extract the response: text
html_doc = r.text

# create a BeautifulSoup object
soup = BeautifulSoup(html_doc)

print(soup.prettify())



# Specify url: url
url3 = 'https://www.python.org/~guido/'

# Package the request, send the request and catch the response: r

r = requests.get(url3)
# Extracts the response as html: html_doc
html_doc2 = r.text

# Create a BeautifulSoup object from the HTML: soup
soup = BeautifulSoup(html_doc2)

# Prettify the BeautifulSoup object: pretty_soup

pretty_soup = soup.prettify()
# Print the response
print(pretty_soup)


# Get the title of Guido's webpage: guido_title
guido_title = soup.title

# Print the title of Guido's webpage to the shell
print(guido_title)

# Get Guido's text: guido_text
guido_text = soup.get_text()

# Print Guido's text to the shell
print(guido_text)

# Find all 'a' tags (which define hyperlinks): a_tags
a_tags = soup.find_all('a')

# Print the URLs to the shell
for link in a_tags:
    print(link.get('href'))


