








l = [{'a': 1, 'v':2}, {'a': 12, 'v':23}, {'a': 21, 'v':22}, {'a': 35, 'v':6}, {'a': 32, 'v':43}]

r = l[:3]
p = l[3:4]
q = l[4:]
print(l)
print(r)
print(p)
print(q)


print(sum([i['a'] for i in l]))
