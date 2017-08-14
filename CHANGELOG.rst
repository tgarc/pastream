Changelog
=========

0.0.7
-----
- enhancements

    + add *--loop* option to the CLI to allow looping playback. Implementing as
      an option to the soundfilestream classes is a little more complicated so
      I'm avoiding that route at the moment.

    + allow empty string as an alternative to null

    + Raise exception when attempting to open stream with RAW playback file if
      any of samplerate/channels/subtype are not specified. This is IMO less
      confusing than assuming the files conforms to the default device
      parameters.
    
    + change prebuffering behavior slightly: only wait until the first write, not
      until the buffer fills up. This should avoid potential long pre-buffer times

    + minor formatting fix in __repr__ when using multiple dtypes and/or
      devices

    + no need to vendor pa_ringbuffer anymore, it's available on pip! (Thanks
      @mgeier !)

    + if a SoundFile *inpf* is passed to a SoundFileInputStream class, it will
      be used to set the stream samplerate/channels.

- bugfixes

    + addresses a bug when BUFFERSIZE < 8192

- API changes & backwards compatibility breaks:

    + Stream and SoundFileStream classes renamed to *DuplexStream
    
    + Swapped assignments of input/output in SoundFileStreams to make it align
      with the usage in the rest of the library. The order of input/output
      arguments from the CLI still stays the same though.

    + remove *allow_drops* parameters. It can be added back at a later point if
      it proves to be a more useful feature
    

0.0.6
-----
- bugixes

    + fix 'null' not properly matching on cmdline

    + chunks: check that portaudio has not been terminated before trying to
      close/stop a stream

- backwards compatibility breaks: 

    + drop allow_runs/XRunError
    + Buffered*Stream -> *Stream
    + *Buffer{Empty,Full} -> Buffer{Empty,Full}

- fix remaining issues with wheel building

- Dropped unused exception classes (PaStreamError, AudioBufferError)
    
- Added prebuffer argument to start() to bypass filling output buffer
  before stream starts
    

0.0.5
-----
- bugfixes

    + Redirect sys.stdout to devnull when '-' is used as the output file stream

    + Specifying multiple ``--file-type`` s at command line fixed

    + ``--format`` now only accepts a single argument

    + ``ringbuffersize_t`` is of a different type for mac platforms; fixed

    + ps.chunks() README example fixed
    
- backwards compatibility breaks: 
    
    + ``frames`` is now a signed value. The behavior previously reserved for
      frames == 0 now is active whenever frames < 0

    + Comma separated arguments are no longer allowed; multiple argument
      options can only be specified by passing them multiple times

    + dropped support for passing a bool for ``pad`` parameter

    + ``-q`` flag for specifying buffersize has been dropped. This is now
      reserved for the new ``--quiet`` option.

- add a loopback test for the pastream app using stdin > stdout

- improvement: ``chunks`` function: make sure that stream is closed properly
  without the performance hit of having an extra yield
    
- new feature: If both ``padding`` and ``frames`` are < 0, padding will be
  added indefinitely
    
- new feature: ``-q/--quiet`` option; this drops the deprecated -q option for
  specifying buffersize

    
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

