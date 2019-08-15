import h5py
import numpy
import random
import re

count = {}
data = []
for line in open('mainsurvey_sqldump.txt'):
    if line.startswith('INSERT INTO "answers"'):
        # INSERT INTO "answers" VALUES(5,2,1267419006.0,75,49,234,'blue');
        m = re.search(r'(\d+),(\d+),(\d+),\'(.*?)\'\)', line)
        r, g, b, name = m.groups()
        r, g, b = int(r), int(g), int(b)
        # print(r, g, b, name)
        count[name] = count.get(name, 0) + 1
        data.append((r, g, b, name))

names = sorted(name for name in count.keys() if count[name] >= 20)
name2i = dict((name, i) for i, name in enumerate(names))
data = [(r, g, b, name) for r, g, b, name in data if name in names]
random.shuffle(data)
labels = numpy.array([name2i[name] for r, g, b, name in data], dtype=numpy.uint32)
rgb = numpy.array([[r, g, b] for r, g, b, name in data], dtype=numpy.uint8)
with h5py.File('colors.hdf5', 'w') as f:
    f.create_dataset('rgb', data=rgb)
    f.create_dataset('labels', data=labels)
