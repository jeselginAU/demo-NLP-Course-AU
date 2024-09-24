import os
import nbformat
from nbconvert import PythonExporter

def convert_notebooks_to_scripts(notebook_dir):
    for root, dirs, files in os.walk(notebook_dir):
        for file in files:
            if file.endswith(".ipynb"):
                notebook_path = os.path.join(root, file)
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    notebook = nbformat.read(f, as_version=4)
                script, _ = PythonExporter().from_notebook_node(notebook)
                script_path = os.path.join(root, file.replace(".ipynb", ".py"))
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(script)

# Change '.' to your project directory containing the Jupyter notebooks
convert_notebooks_to_scripts('.')
