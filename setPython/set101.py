#!/usr/bin/env python
params = {"sep":"\n\n", "end":"\n\n"}

# Creating two sets 
set1 = set(list('kmangokileuyrt')) 
wordlist = 'Mango banana Usuer pretend score cure tap tip toppit going down reef tree'.split() 
set2 = set(list('Mango'))
set3 = set()
set4 = set()

print(set1, set2, set3, set4, wordlist, **params)

# Adding elements to set3 
for i in range(1, 10): 
    set3.add(i) 
  
# Adding elements to set4 
for i in range(34, 56): 
    set4.add(i) 

  
# Union of set3 and set4 
set5 = set3 | set4 # set3.union(set4) 
print("Union of Set3 & Set4: Set5 = ", set5, **params) 
  
# Intersection of set3 and set4 
set6 = set4 & set3 # set3.intersection(set4) 
print("Intersection of Set3 & Set4: Set6 = ", set6, **params) 

  
# Checking relation between set3 and set4 
if set3 > set4: # set3.issuperset(set4) 
    print("Set3 is superset of Set4", **params) 
elif set3 < set4: # set3.issubset(set4) 
    print("Set3 is subset of Set4", **params) 
elif set3 == set4: # set3.issubset(set4) 
    print("Set3 is same as Set4", **params) 
else : # set3 == set4 
    print("Set3 is not the same as Set4", **params) 
  


for word in set2:
    set7 = set(list(word))
    if set5 < set6: # set5.issubset(set6) 
        print(f"{word} can be made from the letter", **params)
    
