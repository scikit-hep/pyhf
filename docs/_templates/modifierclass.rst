{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ name }}
   :members:
   :inherited-members:
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
   .. automethod:: {{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
