from pipette.types import HDFFile, FitsFile, DataFile

class TomoCatFile(HDFFile):
    pass

class RandomCatFile(HDFFile):
    pass

class DiagnosticMapsFile(HDFFile):
    pass

class PhotozPDFFile(HDFFile):
    pass

class ShearCatFile(FitsFile):
    pass

class SACCFile(DataFile):
    suffix = 'sacc'
