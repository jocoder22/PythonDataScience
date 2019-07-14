#!/usr/bin/env python
import re
from colorama import init
init()

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

emails2 = re.compile(r'\w+@\w+\.com')
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
        # print(f'\33[48;5;42m{address} is a valid email address')
        print(f'\033[32m{address} is a valid email address') # green letter
    else:
        print(f'\033[31m{address} is NOT a valid email address') # red coloured letters
        # print(f'\33[48;5;47m\033[31m{address} is NOT a valid email address')  # white background on red letters

