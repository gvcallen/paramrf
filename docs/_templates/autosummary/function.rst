{% if fullname.count('.') <= 1 %}
{% set title = "**" ~ name ~ "** (" ~ fullname ~ ")" %}
{% else %}
{% set title = name %}
{% endif %}
{{ title }}
{{ "=" * title|length }}

.. currentmodule:: {{ module }}

.. autofunction:: {{ objname }}