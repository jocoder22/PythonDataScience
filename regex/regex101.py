#!/usr/bin/env python
import re
from colorama import init
init(autoreset=True)

sp = '\n\n'

pattern =  re.compile(r'w{3}.\w+.com')
email = re.compile(r'\w+@\w+.com')
nonchar = re.compile(r'\W')

text = '''from: www.jugjss569.com, to www.maondgt.com
          cc: www.kildn.com, www.jdldaa.com, ladjda@yahoo.com, poty@hdd.com
          header: check this website, www.viddaf.com, email: 7344dkd@jdas.com
          body: This is to inform you of jane absence today. Jane is not feeling
          well today. Today is Viddaf's birthday too!.
          '''
text2 = """! ' ~  ` ! @ # $ % ^ & * ( ) _ + = / . , < > : ; ' "[]{}\; - '"""

text3 = "{|Please%$remember&to:']scrabble,><this%$$email)for!@security=#}purposes!"

result = pattern.findall(text)
print(result)

emailaddress = email.findall(text)
print(emailaddress)

noncharacter = nonchar.findall(text2)
print(noncharacter)


# metacharacters
# . (dot) => any character

emails2 = re.compile(r'\w+@.+?\.com')
findemail2 = emails2.findall(text)
print(findemail2)

# | or 
using_or = re.compile('Jane|jane')
using_classes = re.compile(r'[Jj]ane')

find_or = using_or.findall(text)
find_class = using_classes.findall(text)
print(find_or, find_class, sep=sp, end=sp)

# classes (square brackets) => select any in the square bracket
coded = re.compile(r'[\W]+')
decode = coded.sub(" ", text3)
print(decode)


# check valid email address => don't start with numbers
valid = re.compile(r'[a-zA-Z]+@\w+\.com')
for address in findemail2:
    if valid.match(address):
        # 033[(forground);(background);(transparency)
        # print(f'\33[48;5;42m{address} is a valid email address')
        print(f'\033[32m{address} is a valid email address') # green letter
    else:
        # print(f'\033[36;45;1m{address} is NOT a valid email address\033[0m') # red coloured letters
        print(f'\33[31;47;1m{address} is NOT a valid email address')  # white background on red letters


html = """
  <body>
    <div class="longbox row">
        <h2>Hi! I am the Base Project</h2>
        <p> This is the last day of work before christmas. I'm so exited to work today. Although the office might be scanty because some staffs are on holiday off, there are lots of funs and good tidings around the office. This is a wonderful day indeed.</p>
        <p> While working on the last day of the week before christmas, sweet melody playing in the background; sweet christmas choral, the quietness and serenity of the day is so gentle and soathing. Oh wonderful christmas again!</p>
    </div>
  </body>

"""

text4 = """
        This is awesome (although new to me) I have to (try to engage) commpletely (don't mind
listening carefully) throughout the demo Good stuffs to learn (I keep learning always!)
        You can beat this (Remember the Music!).
        """

text5 = "This is awesome (although new to me) I have (to try to engage) commpletely"

# Non greedy Match (Lazy Match)
html_r = re.compile(r'<.+?>')
no_html = html_r.sub(" ", html)
print(no_html)

no_html2 = html_r.findall(html)
print(no_html2, end=sp)

speech = re.compile(r'\(.+?\)')
add_speech = speech.findall(text4)
print(add_speech, end=sp)

# Greedy Match
html_r3 = re.compile(r'<.+>')
no_html3 = html_r3.findall(html)
print(no_html3)

## Dont work on multiline text
print('################################')
speech2 = re.compile(r'\(.+\)', flags = re.MULTILINE)
add_speech2 = speech2.findall(text4)
print(add_speech2, end=sp)

add_speech3 = speech2.findall(text5)
print(add_speech3, end=sp)