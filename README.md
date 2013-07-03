Introduction
============
NeXpy provides a high-level python interface to NeXus data contained within a simple GUI. It is designed to provide an intuitive interactive toolbox allowing users both to access existing NeXus files and to create new NeXus-conforming data structures without expert knowledge of the file format.

Installing and Running
======================

```
  python setup.py install
```

To install in an alternate location:

```
  python setup.py install --prefix=/path/to/insallation/dir
```

Prerequisites (must install before you can run)
===============================================

* PySide v1.1.0        http://www.pyside.org/
* iPython v0.13        http://ipython.org/
* numpy,scipy          http://numpy.scipy.org
* matplotlib v1.1.0    http://matplotlib.sourceforge.net    (GUI only)
* hdf5                 http://www.hdfgroup.org
* mxml                 http://www.minixml.org
* nexus                http://www.nexusformat.org
* pyspec               http://pyspec.sourceforge.net    (SPEC reader only)

The following environment variables may need to be set
NEXUSLIB --> /point/to/lib/libNeXus.so
PYTHONPATH --> must include paths to ipython,numpy,scipy,matplotlib if installed in nonstandard place

All of the above are included in the Enthought Python Distribution v7.3.

PyQt4 should also work instead of PySide. NeXpy automatically checks for 
which PyQt variant is available (PySide or PyQt4 - not PyQt). 

To run with the GUI
===================

You can run both from the source location or from the install location. To run
from the source directory, simply execute:

```
scripts/nexpy
```

To run from the install location, add the $prefix/bin directory to your path
(only if you installed outside the python installation), and then run:

```
nexpy
```

Online Help
===========
There is a brief tutorial on using NeXpy and the NeXus Python tree API on the [NeXus wiki](http://wiki.nexusformat.org/NeXpy).