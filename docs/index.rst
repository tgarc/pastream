.. pastream documentation master file, created by
   sphinx-quickstart on Sat Aug 19 22:03:48 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. default-role:: py:obj

.. include:: ../README.rst

.. include:: ../CHANGELOG.rst


..
   Command Line Reference
   ==========================

   .. argparse::
      :module: pastream
      :func: _get_parser
      :prog: pastream
      :nodescription:


API Reference
=============

.. automodule:: pastream
   :members:
   :exclude-members: InputStream, Stream, DuplexStream, OutputStream, RingBuffer

.. autoclass:: RingBuffer
   :inherited-members:

.. autoclass:: Stream
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: InputStream
   :members: chunks, set_sink, record, rxmapping, to_file

.. autoclass:: OutputStream
   :members: set_source, play, txmapping, from_file

.. autoclass:: DuplexStream
   :members: playrec, rxmapping, txmapping, to_file, from_file
