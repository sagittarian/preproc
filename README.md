Preproc
=======
**Preproc** is a library for preprocessing data for machine learning.

## Usage
To install, create a Python virtualenvironment if you want, then:
```
$ git clone https://github.com/sagittarian/preproc.git
$ cd preproc
$ pip install -r requirements.txt
$ pip install .
```

You will need to create a `preproc.preprocess.Preprocess` instance and
supply it with steps to use for preprocessing.  The step classes are in
the `preproc.steps` module.  Step objects can either be passed to the
`preproc.preprocess.Preprocess` constructor, or added after creating the
object by calling the `add` method.

The preprocessing pipeline has two stages, fit and transform.  In the
fit stage, training data is passed through the pipeline, allowing stages
to learn any parameters that depend on the data (such as the average of
the data, etc.).  In the transform stage, new data is transformed
according to the given steps and the parameters learned in the fit
stage.  It is an error to add more steps to the preprocessing pipeline
after it has been fitted.

## Tests
To run the tests:
```
$ pip install -r requirements-dev.txt
$ pytest
```
