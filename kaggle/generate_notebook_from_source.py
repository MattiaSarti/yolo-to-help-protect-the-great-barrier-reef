"""
Generate the competition notebook by assembling all the source code spread
across the different repository files.
"""


from os import getcwd, listdir, pardir
from os.path import join as path_join

from nbformat import write
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


NOTEBOOK_MARKDOWN_HEADERS = (
    """# Mattia Sarti's Notebook\n""" +
    """### The following source code is illustrated [here]""" +
    """(https://github.com/MattiaSarti/""" +
    """yolo-to-help-protect-the-great-barrier-reef)"""
)
NOTEBOOK_PATH = path_join(
    getcwd(),
    'submitted_notebook.ipynb'
)
SOURCE_CODE_DIR = path_join(
    getcwd(),
    pardir,
    'source'
)
SOURCE_FILENAMES_ORDERED_AS_COPIED = [
    'common_constants.py',
    'samples_and_labels.py',
    'model_architecture.py',
    'loss_and_metrics.py',
    'training_and_validation.py',
    # 'inference.py'
]


def create_notebook_assembling_all_source() -> None:
    """
    Generate the competition notebook by assembling all the source code spread
    across the different repository files.
    """
    def read_file_content(file_path):
        """
        Read the whole content of the input file whose path is given.
        """
        with open(file_path, 'r') as file:
            file_content = file.read()
        return file_content

    source_code_files_paths = map(
        lambda name: (
            name, read_file_content(path_join(SOURCE_CODE_DIR, name))
        ),
        SOURCE_FILENAMES_ORDERED_AS_COPIED
    )

    notebook_cells = [
        # appending markdown headers:
        new_markdown_cell(source=NOTEBOOK_MARKDOWN_HEADERS),
        # appending a cell with a code illustration header:
        new_markdown_cell(source='#### Settings'),
        # appending a code cell for the __name__ variable modification so as
        # to run only what required:
        new_code_cell(source="""__name__ = 'main_by_mattia'""")
    ]
    for filename, source_text in source_code_files_paths:
        # appending a cell with a code illustration header:
        notebook_cells.append(
            new_markdown_cell(
                source='#### ' + filename.replace('_', ' ')[:-3].capitalize()
            )
        )
        # appening a cell with the actual code:
        notebook_cells.append(new_code_cell(source=source_text))

    notebook = new_notebook()
    notebook['cells'] = notebook_cells
    write(notebook, NOTEBOOK_PATH)


if __name__ == '__main__':
    create_notebook_assembling_all_source()
