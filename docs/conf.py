# -- Library configuration ---------------------------------------------------

import matplotlib

# ignore matplotlib warnings
matplotlib.set_loglevel("critical")

# -- Generate tables ---------------------------------------------------------

import pathlib
import subprocess


def run_script(name, cwd):
    path = pathlib.Path(cwd)
    if not path.is_dir():
        raise ValueError(f"Path {cwd} does not exist")

    process = subprocess.Popen(
        ["python", name],
        cwd=cwd,
    )
    process.wait()


scripts = [
    ("generate_table.py", "healpix"),
]
for name, cwd in scripts:
    run_script(name, cwd)

# -- Project information -----------------------------------------------------

project = "healpix-geo"
year = "2025"
author = "grid4earth project"

copyright = f"{year}, {author}"

# root toctree document
root_doc = "index"

# -- General configuration ---------------------------------------------------

# enabled extensions
extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "jupyter_sphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- autosummary / autodoc ---------------------------------------------------

autosummary_generate = True
autodoc_typehints = "none"

# -- napoleon ----------------------------------------------------------------

napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False

# -- myst-parser -------------------------------------------------------------

myst_enable_extensions = ["dollarmath"]

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/EOPF-DGGS/healpix-geo",
            "icon": "fa brands fa-square-github",
            "type": "fontawesome",
        },
    ],
    "icon_links_label": "Quick Links",
}
