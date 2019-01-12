import os
from PIL import ImageFilter, Image as img


path = 'C:\\Users\\Jose\\Pictures\\'
os.chdir(path)
print(os.listdir())
'light.jpg' in os.listdir()
myfiles = os.listdir()
myfiles.index('light.jpg')

images2 = img.open('light.jpg')
images2.show()
images2.save('light2.png')

newsize = (450, 450)
for file in os.listdir():
    if file.endswith('.jpg'):
        image = img.open(file)
        fname, fext = os.path.splitext(file)
        image.thumbnail(newsize)
        image.save(f'Newfile/{fname}_resized.{fext}')

images2.rotate(90).save('Newfile/light_r.jpg')
images2.convert(mode='L').save('Newfile/light_l.jpg')
