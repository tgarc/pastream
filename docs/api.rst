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
   :members: chunks, set_sink, record

.. autoclass:: OutputStream
   :members: set_source, play

.. autoclass:: DuplexStream
   :members: playrec
