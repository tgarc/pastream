"""Mock module for Sphinx autodoc."""


class ffi(object):
    NULL = NotImplemented

ffi = ffi()


class FakeLibrary(object):
    paInputOverflow = paInputUnderflow = paOutputOverflow = paOutputUnderflow = NotImplemented

lib = FakeLibrary()
