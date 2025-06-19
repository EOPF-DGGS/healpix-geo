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

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
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
