#!/usr/bin/env python
import re


pattern =  re.compile(r'w{3}.\w+.com')

text = '''from: www.jugjss569.com, to www.maondgt.com
          cc: www.kildn.com, www.jdldaa.com, ladjda@yahoo.com, poty@hdd.com
          header: check this website, www.viddaf.com, email: 7344dkd@jdas.com
          body: This is to inform me
          '''
result = pattern.findall(text)
print(result)