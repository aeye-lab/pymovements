{% for item in data %}
.. toctree::

    :hidden:
    :maxdepth: 2

    {{ item }}

{% endfor %}
