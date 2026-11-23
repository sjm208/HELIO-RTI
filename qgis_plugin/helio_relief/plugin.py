import os

from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction

MENU = "&Helio Relief"


class HelioReliefPlugin:
    """QGIS plugin entry point for Helio Relief."""

    def __init__(self, iface):
        self.iface = iface
        self.dialog = None
        self.action = None
        self.plugin_dir = os.path.dirname(__file__)

    def initGui(self):
        self.action = QAction(QIcon(os.path.join(self.plugin_dir, "icon.png")),
                              "Helio Relief", self.iface.mainWindow())
        self.action.setToolTip("Relief renders, hillshades, measurement and plates for height maps - opens Helio exports")
        self.action.triggered.connect(self.run)
        self.iface.addPluginToMenu(MENU, self.action)
        self.iface.addRasterToolBarIcon(self.action)

    def unload(self):
        self.iface.removePluginMenu(MENU, self.action)
        self.iface.removeRasterToolBarIcon(self.action)
        if self.dialog is not None:
            self.dialog.shutdown()
            self.dialog.deleteLater()
            self.dialog = None

    def run(self):
        if self.dialog is None:
            # Imported lazily so QGIS starts fast even if numpy/scipy are slow to load.
            from .dialog import HelioReliefDialog
            self.dialog = HelioReliefDialog(self.iface.mainWindow(), self.iface)
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
