import os

fname = os.listdir('pic')
fname.sort()

name = []
for item in fname:
    item = item.replace(".png", '')
    item = int(item)
    name.append(item)
name.sort()

for item in name:
	print item
