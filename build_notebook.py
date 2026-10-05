"""Build a fresh, UNEXECUTED notebook from nb_src.txt.

nb_src.txt is the plain-text source of every cell (markdown and code), kept so changes are easy to read in a diff.
Cells are separated by lines '#%% md' and '#%% code'.

    python build_notebook.py                      # writes training_loop_built.ipynb (safe: never touches the executed notebook)
    python build_notebook.py training_loop.ipynb  # overwrite the committed notebook; refuses if it has outputs, unless --force
"""
import re, sys, json, os
import nbformat

args = [a for a in sys.argv[1:] if not a.startswith('--')]
out = args[0] if args else 'training_loop_built.ipynb'
if os.path.exists(out) and '--force' not in sys.argv:
    try:
        has_outputs = any(c.get('outputs') for c in json.load(open(out, encoding='utf-8')).get('cells', []))
    except Exception:
        has_outputs = False
    if has_outputs:
        sys.exit(f'{out} contains executed outputs; refusing to overwrite it with an unexecuted notebook. Add --force to do it anyway.')

src = open('nb_src.txt', encoding='utf-8').read()
parts = re.split(r'^#%% (md|code)\s*$', src, flags=re.M)
nb = nbformat.v4.new_notebook()
for kind, body in zip(parts[1::2], parts[2::2]):
    body = body.strip('\n')
    nb.cells.append(nbformat.v4.new_markdown_cell(body) if kind == 'md' else nbformat.v4.new_code_cell(body))
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbformat.write(nb, out)
print(f'{len(nb.cells)} cells -> {out} (unexecuted; run it to fill in the outputs)')
