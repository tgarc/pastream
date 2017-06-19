Changelog
=========

0.0.4
-----
- bugfix: chunks: overlap was (accidentally) not allowed if chunksize was
  not non-zero. This should be allowed as long as stream.blocksize > 0.

- chunks now supports passing a generic ndarray to ``out`` parameter (without
  having to cast it to a bytes object)

- ``nframes`` renamed to ``frames``

- ``padding`` renamed to ``pad``

- added ``allow_drops`` option to give user the option to ignore
  ``ReceiveBufferEmpty`` error in more atypical use cases

- ``raise_on_xruns`` changed to ``allow_xruns``; inverted behavior

- got rid of undocumented ``keep_alive`` option; the combination of ``allow_drops`` and
  ``pad`` can give the same functionality

- ``--pad`` now can be specified without an argument which just sets pad to
  True

- added autopadding feature: Now if ``frames`` > 0 and ``pad`` == True or pad <
  0, playback will be zero padded out to ``frames``. This is a nice feature for
  the pastream application and SoundFileStream since sometimes you want to add
  extra padding after the file playback.

0.0.3
-----
- command line options for size parameters now accept k/K/m/M suffix

- Backwards compatibility break: multiple argument command line options now
  accept a comma delimited list

- improved SoundFileStream reader writers; nearly zero read/write misses

- bugfix: __repr__ had a bug for certain cases

0.0.2
-----

- Improved SoundFileStream interface: remove sfkwargs; instead format, endian,
  and subtype can be passed directly since they don't collide with any of the
  sounddevice parameters
    
- Updated examples to allow half or full duplex operation. Also accepts subtype
  for RAW files

- chunks() updates
  + better polling behavior greatly decreases read misses
  + now supports generic buffers so numpy is not required
  + added `out` option to allow user to pass a preallocated buffer
  + bugfix: overlap was not overlapping correctly

- MAJOR bugfix: samplerate was not being properly passed up the class chain

- MAJOR bugfix: lastTime was not being properly copied in py_pastream.c so the value
  returned was garbage 

- bugfix: assert_chunks_equal: the 'inframes' buffer was not being allocated
  enough space for when chunksize > blocksize which was causing mismatch
  hysteria

0.0.1
-----
First tenable release

