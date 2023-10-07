"""
# =============================================================================
# Functions in order to visualise 2D and 3D NumPy arrays.
#
# Author: William Hunter
# Copyright (C) 2008, 2015, 2016, 2017 William Hunter.
# =============================================================================
"""

from __future__ import division

import sys
from datetime import datetime
from pylab import axis, close, cm, figure, imshow, savefig, title
from numpy import arange, asarray, hstack
from pyvtk import CellData, LookupTable, Scalars, UnstructuredGrid, VtkData

__all__ = ['create_2d_imag', 'create_3d_geom', 'node_nums_2d', 'node_nums_3d',
'create_2d_msh','create_3d_msh']


def create_3d_geom(x, **kwargs):
    """
    Create 3D geometry from a 3D NumPy array.

    Takes a 3D array as argument and saves it as geometry. Each value in the
    array is represented by a 1x1x1 cube and the colour of each cube is
    determined by the value of the array entry, which must vary between 0.0 and
    1.0. Array entries with values below THRESHOLD are culled from the geometry
    (see source for details). A 'dd-mm-yyyy-HHhMM' timestamp is automatically
    added to the filename unless the function is called with the time='none'
    keyword argument. Default, and only file type, is legacy VTK unstructured
    grid file ('vtk' extension); it does not have to be specified.

    INPUTS:
        x -- K-by-M-by-N array (depth x rows x columns)

    OUTPUTS:
        <filename>.<type>

    OPTIONAL INPUTS (keyword arguments):
        prefix -- A user given prefix for the file name; default is 'topy_3d'.
        filetype -- The visualisation file type, see above.
        iternum -- A number that will be appended after the filename; default
                   is 'nin'.
        time -- If 'none', then NO timestamp will be added.

    EXAMPLES:
        >>> create_3d_geom(x, iternum=12, prefix='mbb_beam')
        >>> create_3d_geom(x)
        >>> create_3d_geom(x, time='none')

    """
    # Set the filename component defaults:
    keys = ['dflt_prefix', 'dflt_iternum', 'dflt_timestamp', 'dflt_filetype']
    values = ['topy_3d', 'nin', '_' + _timestamp(), 'vtk']
    fname_dict = dict(zip(keys, values))
    # Change the default filename based on keyword arguments, if necessary:
    fname = _change_fname(fname_dict, kwargs)
    # Save the domain as geometry:
    _write_geom(x, fname)


# =====================================
# === Private functions and helpers ===
# =====================================
def _change_fname(fd, kwargs):
    # Default file name:
    filename = fd['dflt_prefix'] + '_' + fd['dflt_iternum'] + \
    fd['dflt_timestamp'] + '.' + fd['dflt_filetype']

    # This is not pretty but it works...
    if 'prefix' in kwargs:
        filename = filename.replace(fd['dflt_prefix'], kwargs['prefix'])
    if 'iternum' in kwargs:
        fixed_iternum = _fixiternum(str(kwargs['iternum']))
        filename = filename.replace(fd['dflt_iternum'], fixed_iternum)
    if 'filetype' in kwargs:
        ftype = kwargs['filetype']
        filename = filename.replace(fd['dflt_filetype'], ftype)
    if 'time' in kwargs:
        filename = filename.replace(fd['dflt_timestamp'], '')
    if 'dir' in kwargs:
        dir = kwargs['dir']
        if not  dir[-1] == '/':
            dir = dir + '/'
        filename = dir + filename

    return filename

def _write_geom(x, fname):
    '''
    Determines what geometry format (file type) to create.
    '''
    if fname.endswith('vtk', -3):
        _write_legacy_vtu(x, fname)
    else:
        print ('Other file formats not implemented, only legacy VTK.')
        #_write_vrml2(x, fname) # future

def _write_legacy_vtu(x, fname):
    """
    Write a legacy VTK unstructured grid file.

    """
    # Lower bound value used for pixel/voxel culling, any value below this
    # value won't be plotted. Should be same as VOID's value in 'topology.py'.
    THRESHOLD = 0.001

    # Voxel local points relative to its centre of geometry:
    voxel_local_points = asarray([[-1,-1,-1],[ 1,-1,-1],[-1, 1,-1],[ 1, 1,-1],
                                [-1,-1, 1],[ 1,-1, 1],[-1, 1, 1],[ 1, 1, 1]])\
                                  * 0.5 # scaling
    # Voxel world points:
    points = []
    # Culled input array -- as list:
    xculled = []

    try:
        depth, rows, columns = x.shape
    except ValueError:
        sys.exit('Array dimensions not equal to 3, possibly 2-dimensional.\n')

    for i in range(depth):
        for j in range(rows):
            for k in range(columns):
                if x[i,j,k] > THRESHOLD:
                    xculled.append(x[i,j,k])
                    points += (voxel_local_points + [i,j,k]).tolist()

    voxels = arange(len(points)).reshape(len(xculled), 8).tolist()
    topology = UnstructuredGrid(points, voxel = voxels)
    file_header = \
    'ToPy data, created '\
    + str(datetime.now()).rsplit('.')[0]
    scalars = CellData(Scalars(xculled, name='Densities', lookup_table =\
    'default'))
    vtk = VtkData(topology, file_header, scalars)
    vtk.tofile(fname, 'binary')

def _timestamp():
    """
    Create and return a timestamp string.

    """
    now = datetime.now()
    day = _fixstring(str(now.day))
    month = _fixstring(str(now.month))
    year = str(now.year)
    hour = _fixstring(str(now.hour))
    minute = _fixstring(str(now.minute))
    ts = day + '-' + month + '-' + year + '-' + hour + 'h' + minute
    return ts

def _fixstring(s):
    """
    Fix the string by adding a zero in front if single digit number.

    """
    if len(s) == 1:
        s = '0' + s
    return s

def _fixiternum(s):
    """
    Fix the string by adding a zero in front if double digit number, and two
    zeros if single digit number.

    """
    if len(s) == 2:
        s = '0' + s
    elif len(s) == 1:
        s = '00' + s
    return s

# EOF visualisation.py
