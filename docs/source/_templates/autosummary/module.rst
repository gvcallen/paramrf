{% if fullname.count('.') <= 1 %}
{% set title = "**" ~ name ~ "** (" ~ fullname ~ ")" %}
{% else %}
{% set title = name %}
{% endif %}
{{ title }}
{{ "=" * title|length }}

.. automodule:: {{ fullname }}
   :no-members:
   :no-inherited-members:

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :template: autosummary/module.rst
   :recursive:

{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block classes %}
{% if classes %}
.. rubric:: Classes

.. autosummary::
   :toctree:
   :template: autosummary/class.rst

{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block functions %}
{% if functions %}
.. rubric:: Functions

.. autosummary::
   :toctree:
   :template: autosummary/function.rst

{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}