import os 

def print2(*args):
    for arg in args:
        print(arg, end='\n\n')
  
sp = {"sep":"\n\n", "end":"\n\n"}


print(os.getcwd(), os.listdir(), **sp)

path = r"D:\PythonDataScience"
os.chdir(path)

print(os.getcwd(), os.listdir(), **sp)
print(os.walk(path), **sp) # this create a generator of tuple (dirpath, dirname, filenames)

for index, (dirpath, dirname, filenames) in enumerate(os.walk(path)):
    print(index, (dirpath, dirname, filenames), end='\n\n')


mylist = []
n = 1   
for index, (dirpath, dirname, filenames) in enumerate(os.walk(path)):
    for filename in filenames:
        # fullpath = '/'.join([dirpath, filename])
        fullpath = os.path.join(dirpath, filename)
        print(index, n, fullpath, end='\n\n')
        mylist.append(fullpath)
        n += 1

# print(mylist)
