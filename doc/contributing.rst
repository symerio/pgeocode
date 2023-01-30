Contributing
============

At present ``pgeocode.py`` is a single module to faciliate vendoring,
please add your changes to this file.

Testing
-------

Unit tests can be run with,

.. code::

    pip install pytest pytest-httpserver

.. code::

    pytest

You can also run ``tox`` to run the unit test will all the supported python versions.

Code style
----------

Pgeocode uses black and flake8 for the code style.

To apply them automatically, install `pre-commmit <https://pre-commit.com/#install>`_

.. code::

   pip install pre-commit
   pre-commit install

then fix any style errors that occurs when you commit. You may run it twice to apply
black and isort. It is also possible to run them manually with,

.. code::

   pre-commit run -a
