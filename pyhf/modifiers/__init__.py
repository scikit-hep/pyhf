registry = {}

'''
Consistent add_to_registry() function that handles actually adding thing to the registry.

Raises an error if the name to register for the modifier already exists in the registry.
'''
def add_to_registry(cls, cls_name=None):
  global registry
  cls_name = cls_name if cls_name else cls.__name__
  if cls_name in registry: raise KeyError('The modifier name "{0:s}" is already taken.'.format(cls_name))
  registry[cls_name] = cls

'''
Meta-class for auto-registering modifiers and injecting into a registry.
'''
class IModifier(type):
    def __new__(cls, clsname, bases, attrs):
        newclass = super(IModifier, cls).__new__(cls, clsname, bases, attrs)
        add_to_registry(newclass)
        return newclass

'''
Decorator for registering modifiers. Two ways to use it.

Way 1: automatically determine the name of the modifier using cls.__name__

  >>> @modifiers.modifier
  >>> ... class histosys(object):
  >>> ...   pass

Way 2: pass in a name to use for the modifier

  >>> @modifiers.modifier('mymodifier')
  >>> ... class histosys(object):
  >>> ...   pass
  >>> ...

Should raise error if not passed in one argument (the class to wrap for automatically grabbing the class name; or the string to name the class).

'''
def modifier(*args):
    if len(args) != 1:
        raise TypeError('@modifier takes exactly 1 argument ({0:d} given)'.format(len(args)))

    def _modifier(name):
        def wrapper(cls):
            add_to_registry(cls, name)
        return wrapper

    if callable(args[0]):
        add_to_registry(args[0])
    elif isinstance(args[0], basestring):
        return _modifier(args[0])
    else:
        raise ValueError('@modifier must be given a basestring instance (string, unicode). You gave it {}'.format(type(args[0])))
