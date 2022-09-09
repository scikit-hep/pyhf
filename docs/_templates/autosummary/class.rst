{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

   .. automethod:: __init__

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   {% for item in attributes %}
   .. autoattribute:: {{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% if name == 'numpy_backend' %}
     {% if 'array_type' in members %}
   .. autoattribute:: {{ name }}.array_type
     {% endif %}
     {% if 'array_subtype' in members %}
   .. autoattribute:: {{ name }}.array_subtype
     {% endif %}
   {% endif %}
   {% endblock %}

   {% block methods %}

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   {% for item in members %}
   {% if item not in attributes and item not in inherited_members and not item.startswith('__') and item not in ['array_type', 'array_subtype'] %}
   .. automethod:: {{ name }}.{{ item }}
   {% endif %}
   {%- endfor %}

   {% endif %}
   {% endblock %}
