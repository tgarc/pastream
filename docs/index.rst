.. pastream documentation master file, created by
   sphinx-quickstart on Sat Aug 19 22:03:48 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. default-role:: py:obj

.. include:: ../README.rst

.. default-role::


API Reference
=============

.. automodule:: pastream
   :members:
   :exclude-members: InputStream, _StreamBase

.. autoclass:: _StreamBase
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: InputStream
   :members: chunks

.. autoclass:: RingBuffer
   :inherited-members:

.. include:: ../CHANGELOG.rst

.. only:: html

   Index
   -----

   :ref:`genindex`
