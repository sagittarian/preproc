Serializing
===========

The first thing we need to do is have a way to define how to serialize
and deserialize objects, given that we have the proper class.  This can
be achieved by defining an abstract class that defines a serialize
method and an deserialize class method, and having all the classes that
need to be serialized and deserialized implement the two methods.

The serialize method should return an object that is JSON serializable,
and deserialize class method should take an object previously returned
by serialize and recreate the Step or Preprocess instance.

Versioning
==========

Some changes to the interface of a step or of the processing pipeline
could be handled in a backward compatible manner.  For example, if a
parameter is added that has a default value with the same behavior as
the original version (without the parameter), then nothing special needs
to be done.

In order to be able to continue loading and using any previously
serialized versions of steps and the pipeline itself, it will be
necessary to keep some version of every class that can handle every
version of every object, and a central way of determining what classes
are available and what they can handle.  Each class would have a name
(by default the class name itself) and a version,

As far as the registration of which classes are available, this could be
done with a metaclass (Python 3.6 added a mechanism for doing this
without a metaclass), but it would be better to avoid fancy metaclasses
if possible, and instead I believe a simple class decorator which would
take all the version information for the class should be clear and
sufficient.

If a class is replaced by a new class that is not backward-compatible,
the old class will have to remain in order to load old serialized
instances.  If it is possible to upgrade the old serialized instances,
the old class should be updated to return the upgraded version, so that
if the instance is reserialized, it will use the new upgraded version
rather than the old version.
