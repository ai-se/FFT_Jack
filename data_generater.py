import os
from random import randint

def get_m():
    n = -1
    while n in seen:
        n = randint(1, movie_cnt)
    seen.add(n)
    return n

s_id = 138493
review_cnt = 145
movie_cnt = 131262
counter, cache_size = 0, 1000

cwd = os.getcwd()
csv_path = os.path.join(cwd, "100x_large.csv")
fd = open(csv_path, 'w')

rows = []
for id in xrange(s_id*100):
    seen = {-1}
    for j in range(review_cnt):
        out = map(str, [id, get_m(), randint(1, 5), counter])
        counter += 1
        rows.append(",".join(out))
        if counter % cache_size == 0:
            fd.write('\n'.join(rows))
            rows = []