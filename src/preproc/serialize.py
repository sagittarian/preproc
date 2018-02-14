from collections import defaultdict


class Serializable(object):
    """Interface for classes that can be serialized and deserialized."""

    # An iterable of attributes that should be saved and loaded, used by
    # the default implementation
    attributes = ()

    def serialize(self):
        """Return a JSON-serializable object representing this instance.

        The default implementation simply serializes each of the
        attributes listed in the object's 'attributes' field and returns
        a dictionary with the keys 'attributes', 'version', and 'name'.

        """
        def maybe_serialize(item):
            return item.serialize() if hasattr(item, 'serialize') else item

        data = dict(attributes={}, version=self.version, name=self.name)
        for attr in self.attributes:
            val = getattr(self, attr)
            val = maybe_serialize(val)
            try:
                val = [maybe_serialize(item) for item in val]
            except TypeError:
                pass
            data['attributes'][attr] = val
        return data

    @classmethod
    def deserialize(cls, data, version):
        """Recreate the instance from the given data and the version registry.

        The default implementation simply pases the 'attributes' field
        of the given data to the class's constructor and returns the
        instance.

        """
        return cls(**data['attributes'])


class version(object):
    """Decorator class to tie version information to a class.

    The version.deserialize method can be used to recreate
    an instance given the serialized data.

    Args:
        version: (required) The version the decorated class represents.
        name: (optional) A name to identify this class, by default
            simply the class name.
        handles: (optional) An iterable of (name, minver, maxver) tuples
            specifying what versions of what classes the decorated class
            can deserialize.  Using None for maxver indicates that the
            class's version (the version parameter) should be used as
            the max version.  If this is missing, the class will only
            be able to deserialize data with exactly the same name
            and version.

    Example usage:

        @version(3, handles=[('Log', 1, None)])
        class Log(Step):
            ...

    """

    registry = defaultdict(list)

    def __init__(self, version, name=None, handles=()):
        self.version = version
        self.name = name
        self.handles = handles

    def __call__(self, cls):
        cls.version = self.version
        cls.name = self.name or cls.__name__
        cls.handles = self.handles

        handles = list(cls.handles)
        if not handles:
            handles.append((cls.name, cls.version, cls.version))
        for (name, minver, maxver) in handles:
            if maxver is None:
                maxver = cls.version
            self.registry[name].append((minver, maxver, cls))
            # sort first by the minimum version ascending, then by the
            # maximum version ascending
            self.registry[name].sort(key=lambda x: (-x[0], -x[1]))
        return cls

    @classmethod
    def deserialize(cls, data):
        version = data['version']
        name = data['name']
        for (minver, maxver, clsobj) in cls.registry[name]:
            if minver <= clsobj.version <= maxver:
                return clsobj.deserialize(data, cls)
        raise TypeError('No class found to deserialize version '
                        '{} of class {!r}'.format(version, name))
