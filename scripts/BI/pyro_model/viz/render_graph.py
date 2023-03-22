import pandas as pd
import matplotlib
from itertools import chain
import sys
import subprocess
from math import isnan

args = sys.argv[1:]
fname = args[0]

df = pd.read_csv(f'{fname}.csv')
with open(f'/tmp/{fname.split("/")[-1]}.gv', 'w') as f:
    f.write('digraph G {\n')
    for s in set(chain(df.s, df.d)):
        f.write(f'\t{s} [height=0.05, width=0.05, fontsize=8];\n')
    for s, d, g, e in zip(df.s, df.d, df.grn, df.exp):
        if isnan(e):
            continue
        if g == 0:
            color = matplotlib.colors.to_hex([0.73, 0, 0.73, e],
                keep_alpha=True)
        else:
            color = matplotlib.colors.to_hex([0, 0.73, 0, e],
                keep_alpha=True)
        color = '"' + color + '"'
        f.write(f"\t{s} -> {d} [arrowsize=0.25, color={color}];\n")
    f.write('}')

subprocess.call(f'dot -Tpdf -Kfdp /tmp/{fname.split("/")[-1]}.gv -o {fname}.pdf', shell=True)
subprocess.call(f'rm /tmp/{fname.split("/")[-1]}.gv', shell=True)
subprocess.call(f'open {fname}.pdf', shell=True)
