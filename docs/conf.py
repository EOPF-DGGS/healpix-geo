# -- Library configuration ---------------------------------------------------

import matplotlib

# ignore matplotlib warnings
matplotlib.set_loglevel("critical")

# -- Generate tables ---------------------------------------------------------

import pathlib
import subprocess


def run_script(name, outpath):
    status = subprocess.run(["python", name, outpath])
    if status.returncode != 0:
        raise RuntimeError(f"script {name} failed to run")


script_root = pathlib.Path("scripts").absolute()
docs_root = pathlib.Path.cwd()
scripts = [
    ("generate_healpix_levels_table.py", "healpix/healpix_levels_table.md"),
    ("generate_latitude_diff_table.py", "latitude_diff_table.md"),
    ("generate_latitude_graphic.py", "ellipsoidal_latitudes.png"),
]
for name, outpath in scripts:
    run_script(script_root.joinpath(name), docs_root.joinpath(outpath))

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
