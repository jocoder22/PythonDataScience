import os
from PIL import Image as img

path = 'C:\\Users\\Jose\\Pictures\\'
os.chdir(path)
print(os.listdir())
'light.jpg' in os.listdir()
myfiles = os.listdir()
myfiles.index('light.jpg')

image1 = img.open('light.jpg')
image1.show()
image1.save('light2.png')

for file in os.listdir():
    if file.endswith('.jpg'):
        image = img.open(file)
        fname, fext = os.path.splitext(file)
        image.save(f'Newfile/{fname}.png')
