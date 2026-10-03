"""Builds training_loop.ipynb from nb_src.txt (cells separated by '#%% md' / '#%% code')."""
import re, nbformat
src = open('nb_src.txt').read()
parts = re.split(r'^#%% (md|code)\s*$', src, flags=re.M)
nb = nbformat.v4.new_notebook()
for kind, body in zip(parts[1::2], parts[2::2]):
    body = body.strip('\n')
    nb.cells.append(nbformat.v4.new_markdown_cell(body) if kind == 'md' else nbformat.v4.new_code_cell(body))
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbformat.write(nb, 'training_loop.ipynb')
print(len(nb.cells), 'cells')
