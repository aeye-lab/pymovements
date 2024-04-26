# Copyright (c) 2022-2024 The pymovements Project Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import importlib.resources
import inspect
import os
import sys
from subprocess import CalledProcessError
from subprocess import run

from pybtex.plugin import register_plugin
from pybtex.style.formatting.plain import Style as PlainStyle
from pybtex.style.labels import BaseLabelStyle

# add relative source path to python path
sys.path.insert(0, os.path.abspath('src'))
sys.path.insert(0, os.path.dirname(os.path.abspath('src')))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath('src'))))


# -- Project information -----------------------------------------------------

project = 'pymovements'
copyright = '2022-2023 The pymovements Project Authors'
author = 'The pymovements Project Authors'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.linkcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.extlinks',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinx_favicon',
    'sphinx_mdinclude',
    'sphinxcontrib.datatemplates',
    'sphinxcontrib.bibtex',
    'nbsphinx',
]


def config_inited_handler(app, config):
    os.makedirs(os.path.join(app.srcdir, app.config.generated_path), exist_ok=True)


def setup(app):
    app.add_config_value('REVISION', 'master', 'env')
    app.add_config_value('generated_path', '_generated', 'env')
    app.connect('config-inited', config_inited_handler)


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# exclude_patterns = []


copybutton_prompt_text = r'>>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: '
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = '\\'
copybutton_here_doc_delimiter = 'EOT'


# -- Options for autosummary -------------------------------------------------
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = False
add_module_names = True

# -- Options for HTML output -------------------------------------------------

html_theme = 'pydata_sphinx_theme'

html_static_path = ['_static']

html_css_files = [
    'custom.css',
]

html_theme_options = {
    'navigation_with_keys': False,
    'external_links': [
        {
            'name': 'Contributing',
            'url': 'https://github.com/aeye-lab/pymovements/blob/main/CONTRIBUTING.md',
        },
    ],
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/aeye-lab/pymovements',
            'icon': 'fa-brands fa-github',
        },
    ],
    'logo': {
        'image_light': 'https://raw.githubusercontent.com/aeye-lab/pymovements/main/docs/source/_static/logo.svg',  # noqa: E501
        'image_dark': 'https://raw.githubusercontent.com/aeye-lab/pymovements/main/docs/source/_static/logo.svg',  # noqa: E501
    },
}

# -- Options for favicons

favicons = [
    {'href': 'icon.svg'},
]

# -- Options for juypter notebooks

nbsphinx_execute = 'auto'


# -- Options for BibTeX ------------------------------------------------------
bibtex_bibfiles = ['bibliography.bib']
bibtex_default_style = 'author_year_style'
bibtex_reference_style = 'author_year'


class AuthorYearLabelStyle(BaseLabelStyle):
    def format_labels(self, sorted_entries):
        for entry in sorted_entries:
            yield f'[{entry.persons["author"][0].last_names[0]} et al., {entry.fields["year"]}]'


class AuthorYearStyle(PlainStyle):
    default_label_style = AuthorYearLabelStyle


register_plugin('pybtex.style.formatting', 'author_year_style', AuthorYearStyle)


def getrev():
    try:
        revision = run(
            ['git', 'describe', '--tags', 'HEAD'],
            capture_output=True,
            check=True,
            text=True,
        ).stdout[:-1]
    except CalledProcessError:
        revision = 'main'

    return revision


REVISION = getrev()

extlinks = {
    'repo': (
        f'https://github.com/aeye-lab/pymovements/blob/{REVISION}/%s',
        '%s',
    ),
}

LINKCODE_URL = (
    f'https://github.com/aeye-lab/pymovements/blob/{REVISION}'
    '/src/{filepath}#L{linestart}-L{linestop}'
)


# revised from https://gist.github.com/nlgranger/55ff2e7ff10c280731348a16d569cb73
def linkcode_resolve(domain, info):
    if domain != 'py' or not info['module']:
        return None

    modname = info['module']
    topmodulename = modname.split('.')[0]
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    try:
        modpath = importlib.resources.require(topmodulename)[0].location
        filepath = os.path.relpath(inspect.getsourcefile(obj), modpath)
        if filepath is None:
            return
    except Exception:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        return None
    else:
        linestart, linestop = lineno, lineno + len(source) - 1

    return LINKCODE_URL.format(filepath=filepath, linestart=linestart, linestop=linestop)
