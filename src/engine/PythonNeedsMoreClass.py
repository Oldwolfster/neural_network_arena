class MoreClass:
    """
    In a world of dynamically added attributes, one class dared to stand firm.
    Introducing GetSomeClass, where your Python objects have to earn their attributes. No freeloaders allowed!

    Introducing ClassAct: In a world where Python lets classes go wild with arbitrary
    attributes, one class decided to bring order. This is not your average class—
    it’s strictly for those who demand structure, integrity, and yes... a little class.
    #I'm not sure but i think this was a way around all the self. needed in a class
    """
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, 'allowed_attributes'):
            # Automatically derive allowed attributes from __init__ arguments
            cls.allowed_attributes = set(cls.__init__.__code__.co_varnames[1:])  # Exclude 'self'

    def __setattr__(self, name, value):
        if name not in self.allowed_attributes:
            raise AttributeError(f"Cannot add new attribute '{name}' to {self.__class__.__name__}")
        super().__setattr__(name, value)
