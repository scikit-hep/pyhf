:github_url: https://github.com/scikit-hep/pyhf/blob/master/{{module | replace(".", "/") }}

{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ name }}
   :show-inheritance:

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   {% for item in attributes %}
   .. autoattribute:: {{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   {% for item in methods %}
   {% if item not in inherited_members %}
   .. automethod:: {{ name }}.{{ item }}
   {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
