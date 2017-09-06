from output import xyz
import h5py

class Writer(xyz.Writer):
    # ['atomicNumbers', 'bondOrders', 'bonds', 'coordinates', 'hydrogenBonds', 'masses', 'origin', 'otherCoords', 'title', 'version']
    ext = '.h5mol'

    def _writehead(self,fh):
        return

    def _writezmat(self,fh):
        anums = []
        coords = []
        masses = []
        for _atom in self.parser.zmat:
            anums.append(_atom['number'])
            coords.append(_atom['position'])
            masses.append(_atom['mass'])
         
        #<HDF5 dataset "atomicNumbers": shape (34,), type "<i8">
        #<HDF5 group "/bondOrders" (36 members)>
        #<HDF5 group "/bonds" (36 members)>
        #<HDF5 dataset "coordinates": shape (34, 3), type "<f8">
        #<HDF5 group "/hydrogenBonds" (0 members)>
        #<HDF5 dataset "masses": shape (34,), type "<f8">
        #<HDF5 dataset "origin": shape (3,), type "<f8">
        #<HDF5 dataset "otherCoords": shape (0, 34, 3), type "<f8">
        #<HDF5 dataset "title": shape (), type "|O">
        #<HDF5 dataset "version": shape (), type "|O">
        return

    def _writetail(self,fh):
        return
