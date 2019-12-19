#!/usr/bin/env python
params = {"sep":"\n\n", "end":"\n\n"}

# Creating two sets 
set1 = set(list('kmangokileuyrt')) 
set2 = set(list('Mango banana Usuer pretend score cure tap tip toppit going down reef tree').split()) 
set3 = set()
set4 = set()
# Adding elements to set1 
for i in range(1, 10): 
    set3.add(i) 
  
# Adding elements to set2 
for i in range(34, 56): 
    set4.add(i) 

  
# Union of set1 and set2 
set5 = set3 | set4 # set1.union(set2) 
print("Union of Set3 & Set4: Set5 = ", set5, **params) 
  
# Intersection of set1 and set2 
set6 = set4 & set3 # set1.intersection(set2) 
print("Intersection of Set3 & Set4: Set6 = ", set6, **params) 

  
# Checking relation between set3 and set4 
if set3 > set4: # set3.issuperset(set4) 
    print("Set3 is superset of Set4", **params) 
elif set3 < set4: # set3.issubset(set4) 
    print("Set3 is subset of Set4", **params) 
else : # set3 == set4 
    print("Set3 is same as Set4", **params) 
  


for word in set2:
    set7 = set(list(word))
    if set5 < set6: # set5.issubset(set6) 
        print(f"{word} can be made from the letter", **params)
    
