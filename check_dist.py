import pandas as pd
from collections import Counter

df = pd.read_csv('data/train.csv', encoding='utf-8')
bands = df['Overall'].dropna().tolist()
c = Counter(bands)
total = len(bands)
print('Band distribution in train.csv (Overall column):')
print(f'{"Band":<8} {"Count":>6} {"Bar":<45} {"Pct":>6}')
print('-' * 70)
for b in sorted(c):
    bar = '#' * c[b]
    pct = c[b] / total * 100
    print(f'  {b:<6} {c[b]:>6}  {bar:<45} {pct:>5.1f}%')
print('-' * 70)
print(f'Total: {total}')
print(f'Mean:  {sum(bands)/total:.2f}')
print(f'Min:   {min(bands)}')
print(f'Max:   {max(bands)}')
