"""
Module to read in a TIFF file and convert it to NeXus.

Each importer needs to layout the GUI buttons necessary for defining
the imported file and its attributes and a single module, get_data,
which returns an NXroot or NXentry object. This will be added to the
NeXpy tree.
"""
from nexpy.gui.pyqt import QtWidgets

from nexpy.gui.importdialog import BaseImportDialog
from nexusformat.nexus import *
from nxremote.pyro.nxfileremote import nxloadremote


filetype = "Remote File"
remote_file = (
    '/home/bessrc/sharedbigdata/data1/osborn-2014-1/bfap00/bfap00_170k.nxs')


class ImportDialog(BaseImportDialog):
    """Dialog to open remote data file on a proxy server"""

    def __init__(self, parent=None):

        super(ImportDialog, self).__init__(parent)

        self.layout = QtWidgets.QVBoxLayout()
        self.uri_box = QtWidgets.QLineEdit('PYRO:rosborn@localhost:8801')
        self.file_box = QtWidgets.QLineEdit(remote_file)
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(QtWidgets.QLabel('URI:'), 0, 0)
        grid.addWidget(QtWidgets.QLabel('File Path:'), 1, 0)
        grid.addWidget(self.uri_box, 0, 1)
        grid.addWidget(self.file_box, 1, 1)
        self.layout.addLayout(grid)
        self.layout.addWidget(self.close_buttons())

        self.setLayout(self.layout)

        self.setMinimumWidth(400)

        self.setWindowTitle('Open Remote File')

    def get_data(self):
        uri = self.uri_box.text()
        file_name = self.file_box.text()
        root = nxloadremote(file_name, uri)

        name = self.mainwindow.treeview.tree.get_name(file_name)
        root.nxname = name
        return root
