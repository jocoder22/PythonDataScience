#!/usr/bin/env python
import re


pattern =  re.compile(r'w{3}.\w+.com')
email = re.compile(r'\w+@\w+.com')
nonchar = re.compile(r'\W')

text = '''from: www.jugjss569.com, to www.maondgt.com
          cc: www.kildn.com, www.jdldaa.com, ladjda@yahoo.com, poty@hdd.com
          header: check this website, www.viddaf.com, email: 7344dkd@jdas.com
          body: This is to inform me 
          '''
text2 = """! ' ~  ` ! @ # $ % ^ & * ( ) _ + = / . , < > : ; ' "[]{}\; - '"""

result = pattern.findall(text)
print(result)

emailaddress = email.findall(text)
print(emailaddress)

noncharacter = nonchar.findall(text2)
print(noncharacter)


# metacharacters
# . (dot) => any character

emails2 = re.compile(r'\w+@.+com')
findemail2 = emails2.findall(text)
print(findemail2)
