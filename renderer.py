# -*- coding: utf-8 -*-
"""
@author: raph988

In this script a partial Pyside2 based GUI used to dynamically display datas.
Not really usefull, the original purpose was to create a GUI to display 3D data.
"""

# from PySide2.QtWidgets import QApplication, QWidget, QFrame, QGroupBox, QComboBox, QPushButton, QGridLayout, QCheckBox, QHBoxLayout,QVBoxLayout, QTabWidget, QMessageBox, QMainWindow, QTextEdit, QLineEdit, QLabel, QInputDialog, QProgressDialog, QStyle


from PySide2 import QtCore, QtUiTools, QtWidgets, QtGui
import visvis as vv
#from visvis.core import Wibject
import numpy as np
import os, cv2
from subprocess import call
from importlib import import_module

qtFiles_path = os.path.normpath("Qt/")
compiler_fileName = "compile_ui.bat"
ui_fileName = "figurectrl.ui"
compiler_path = os.path.join(qtFiles_path, compiler_fileName)
ui_path = os.path.join(qtFiles_path, ui_fileName)

mode2d = 0
mode3d = 1

global app
app = vv.use('pyside2')


class BoxDelim(vv.base.Wobject):
    def __init__(self, parent=None):
        super(BoxDelim, self).__init__(None)
        


class BoxVisvis(QtWidgets.QMainWindow):
    
    onFocus = QtCore.Signal()
    
    def __init__(self, parent=None, title = None):
        super(BoxVisvis, self).__init__(None)
        self.frame = QtWidgets.QFrame()
        self.setCentralWidget(self.frame)
        self.mainLayout = QtWidgets.QHBoxLayout()
        self.frame.setLayout(self.mainLayout)
        if title is not None:
            self.setWindowTitle(title)
        
        self.menu = QtWidgets.QMenu(self)
        screenshotAction = QtWidgets.QAction("Sceenshot", self, 
                                         shortcut=QtGui.QKeySequence.Save, 
                                         statusTip="Save the current view in the specified folder.",
                                         triggered=self.screenshot)
        self.menu.addAction(screenshotAction)
        self.statMessage = QtWidgets.QLabel()
        self.statMessage.setAlignment(QtCore.Qt.AlignRight)
        self.statusBar().addWidget(self.statMessage, 1)
        
            
        Figure = app.GetFigureClass()
        self.fig = Figure(self)
        self.ax = vv.gca()
        self.ax.daspectAuto = True
        self.ax.description = 'Axes'
        self.ax.showGrid = True
        self.ax.showMinorGrid = True
        self.camera_backup = self.ax.camera
        # vv.axis('ij')
        # equivalent to :
        self.ax.daspect = 1,-1,1
        
        self.drawn = []
        self.data_drawn = []
        self.overfliedWobj = None
        self.colors = "mrbcygwk"
        self.last_col_used_idx = 0
        
        self.mainLayout.addWidget(self.fig._widget)
        self.scaleFactor = 1.2
        # self.resize(560*sf, 420*sf)
        
        self.fig.eventAfterDraw.Bind(self.newObjDrawn)
        self.installEventFilter(self)
        self.fontImage = None
        self.data = []
        self.data_backup = []
        
        self.show()
            
    
    
    def setAsActive(self):
        vv.figure(self.fig.nr)
        self.onFocus.emit()
    
    def setTitle(self, title):
        self.setWindowTitle(title)
    
    def set_Z_Image(self, z, slider_min_max):
        if self.fontImage is not None:
            _min, _max = slider_min_max
            z = z/_max
            self.fontImage._trafo_trans.dz = z
#            self.fontImage.Refresh()
            self.repaintData(z)
    
    def repaintData(self, z):
        for i in range(0, len(self.data_backup), 1):
            d = self.data_drawn[i]
            backup_verts = self.data_backup[i][0]
            backup_cols = self.data_backup[i][1]
            
            newData = backup_verts.copy()
#            upper = np.where( backup_cols >= z)
#            lower = np.where( backup_cols < z)
            newData[np.where(newData < z)] = z
#            for i in range(0, len(newData), 1):
#                if newData[i][2] < z:
#                    newData[i][2] = z
            
#            d._SetClim(vv.Range(z, 1))
#            d.SetValues(newData)
            d.SetVertices(newData)
        
    def set_xlim(self, xlim):
        _, ylim, zlim = vv.GetLimits()
        vv.SetLimits(xlim, ylim, zlim)
    def set_ylim(self, ylim):
        xlim, _, zlim = vv.GetLimits()
        vv.SetLimits(xlim, ylim, zlim)
    def set_zlim(self, zlim):
        xlim, ylim, _ = vv.GetLimits()
        vv.SetLimits(xlim, ylim, zlim)
        
    def setImageBackground(self, im):
        self.fontImage = vv.imshow(im)
    
    def addData(self, data):
        self.data.append(data)
    
        s = vv.surf(data)
        self.data_drawn.append(s)
        self.data_backup.append((s._vertices.copy(), s._values.copy()))
        return s
    
    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.Type.WindowActivate:
            self.setAsActive()
        return 0
    
    def newObjDrawn(self, event):
        for obj in self.ax.wobjects:
            if isinstance(obj, vv.Wobject) and not isinstance(obj, vv.core.axises.BaseAxis) and obj not in self.drawn:
                self.drawn.append(obj)
                self.setupObj(obj)
    
    def setupObj(self, obj):
        obj.eventEnter.Unbind()
        obj.eventEnter.Bind(self.entering)
        obj.eventLeave.Unbind()
        obj.eventLeave.Bind(self.leaving)
        obj.eventMotion.Unbind()
        obj.eventMotion.Bind(self.picker)
    
    def picker(self, event):
        # event.owner in Visvis equivalent to self.sender() in Qt event system
        
        if self.overfliedWobj is not None:
            txt = "(%i, %i) : %3.2f, %3.2f" % (event.x, event.y, event.x2d, event.y2d)    
            self.statMessage.setText(txt)
    
    def entering(self, event):
        self.overfliedWobj = event.owner
    
    def leaving(self, event):
        self.overfliedWobj = None
        self.statMessage.setText("")
    
    def contextMenuEvent(self, event):
        self.menu.exec_(event.globalPos())
    
    def screenshot(self, fileName = None):
        """
        Opens a dialog to allow user to choose a directory
        """
        if fileName is None:
            fileName,_ = QtWidgets.QFileDialog.getSaveFileName(self, "Save screenshot", os.getcwd(), "Image (*.png)")
            if fileName == '':
                return
        vv.screenshot(fileName, vv.gcf(), sf=1, bg='w')
    
    def closeEvent(self, event):
        """
        Catch close event to confirm exit and also close viewer properly.
        """
        try:
            self.fig.Destroy()
        except:
            pass


class BoxControl(QtWidgets.QMainWindow):
    
    onClose = QtCore.Signal()
    
    def __init__(self, parent=None, qtFiles_path=qtFiles_path, ui_fileName=ui_fileName):
        super(BoxControl, self).__init__(parent)
        
        ui_path = os.path.join(qtFiles_path, ui_fileName)
        ret = 2
        if os.path.exists(ui_path):
            ret = call('pyside-uic "'+ui_path+'" -o "'+ui_fileName.replace('.ui','.py')+'"')
        if ret == 2 :
            self.ui = QtUiTools.QUiLoader().load(ui_path)
        else:
#            module_name = ui_fileName.replace('.ui','')
#            module = import_module(module_name)
#            self.ui = getattr(module, 'Ui_MainWindow')()
            from figurectrl import Ui_MainWindow
            self.ui = Ui_MainWindow()
            self.ui.setupUi(self)
        
        self.ui.b_resetCamera.clicked.connect(self.resetCamera)
        self.ui.b_moveCam_Up.clicked.connect(self.moveCameraUp)
        self.ui.b_moveCam_Down.clicked.connect(self.moveCameraDown)
        self.ui.b_moveCam_Right.clicked.connect(self.moveCameraRight)
        self.ui.b_moveCam_Left.clicked.connect(self.moveCameraLeft)
        self.ui.cb_linkCameras.stateChanged.connect(self.cameraLinkChanged)
        self.ui.z_slider.valueChanged.connect(self.zValueChanged)
#        self.ui.z_slider.sliderReleased.connect(self.zSliderReleased)
        
        
        vv.settings.defaultRelativeFontSize = 1
        self.mBoxes = []
        self.show()
        
        
    def setTitle(self, title):
        self.getActiveBox().setTitle(title)
    
    def addFigure(self, title=None):
        newFigure = BoxVisvis(title=title)
        self.mBoxes.append(newFigure)
        # connection to catch which widget is active
        newFigure.onFocus.connect(self.activeBoxChanged)
        newFigure.setAsActive()
        
        if len(self.mBoxes) > 1:
            self.ui.cb_linkCameras.setEnabled(True)
            
        
        
    def activeBoxChanged(self):
        sender = self.sender()
        self.activeBox = sender

    def setGridState(self, hide = False, boxIndex = None):
        if boxIndex is None:
            for b in self.mBoxes:
                b.ax.showGrid= not hide

    def setActiveBox(self, index):
        self.mBoxes[index].setAsActive()
        
    def getActiveBox(self):
        if not hasattr(self, 'activeBox') or self.activeBox is None: self.addFigure()
        return self.activeBox

    def zValueChanged(self, z):
        self.activeBox.set_Z_Image(z, (self.ui.z_slider.minimum(), self.ui.z_slider.maximum()) )
        
    def zSliderReleased(self):
        self.activeBox.repaintDatas(self.ui.z_slider.value()/100)


    def cameraLinkChanged(self, state):
        if state == QtCore.Qt.CheckState.Checked:
            cam = vv.cameras.ThreeDCamera()
            for f in self.mBoxes:
                f.ax.camera = cam
        else:
            properties = vv.gca().camera.GetViewParams()
            for f in self.mBoxes:
                f.ax.camera = f.camera_backup
                f.ax.camera.SetViewParams(properties)
            
            
    def moveCameraUp(self):
        vv.gca().camera.elevation = 90.01
        vv.gca().camera.azimuth = 0.01
        
    def moveCameraDown(self):
        vv.gca().camera.elevation = -90.01
        vv.gca().camera.azimuth = 180.01
        
    def moveCameraLeft(self):
        vv.gca().camera.elevation = 0.01
        vv.gca().camera.azimuth = -90.01
        
        
    def moveCameraRight(self):
        vv.gca().camera.elevation = 0.01
        vv.gca().camera.azimuth = 90.01
        

    def resetCamera(self):
        # self.activeBox.ax.camera.Reset()
        # equivalent to :
        vv.gca().camera.Reset()
        
    def clearView(self):
        # self.activeBox.fig.Clear()
        # equivalent to :
        # vv.gcf().Clear()
        # equivalent to :
        vv.clf()
#        self.mBoxes.index()

        
    def closeEvent(self, event):
        """
        Catch close event to confirm exit and also close viewer properly.

        """
#        YesOrNo = self.messageBox.question(self, "Message","Exit ?", QtGui.QMessageBox.Yes| QtGui.QMessageBox.No)
#        if YesOrNo == QtGui.QMessageBox.Yes:
#            self.viewer.close()
#            event.accept()
#        else:
#            event.ignore()
        for b in self.mBoxes:
            try:
                b.close()
            except:
                pass
        

        
class Viewer2D(QtWidgets.QMainWindow):
    
    onClose = QtCore.Signal()
    
    def __init__(self, parent=None):
        self.getContext()
        super(Viewer2D, self).__init__(parent)
        
        vv.settings.defaultRelativeFontSize = 1
        self.mBoxes = []
        self.activeBox = None
    
    def getContext(self):
        global app
        if app is None:
            app = vv.use('pyside2')
        return app._GetNativeApp()
    
    def show(self):
        app.Run()
    
    def setTitle(self, title):
        self.getActiveBox().setTitle(title)
    
    def addFigure(self, title=None):
        newFigure = BoxVisvis(title=title)
        self.mBoxes.append(newFigure)
        self.setActiveBox(len(self.mBoxes)-1)
        # newFigure.setAsActive()
        
        
    
    def setGridState(self, hide = False, boxIndex = None):
        if boxIndex is None:
            for b in self.mBoxes:
                b.ax.showGrid= not hide
                
    def setActiveBox(self, index):
        self.mBoxes[index].setAsActive()
        self.activeBox = self.mBoxes[index]
        
    def getActiveBox(self):
        if not hasattr(self, 'activeBox') or self.activeBox is None: self.addFigure()
        return self.activeBox
    
    def resetCamera(self):
        # self.activeBox.ax.camera.Reset()
        # equivalent to :
        vv.gca().camera.Reset()
    
    def clearView(self):
        # self.activeBox.fig.Clear()
        # equivalent to :
        # vv.gcf().Clear()
        # equivalent to :
        vv.clf()
    
    def closeEvent(self, event):
        """
        Catch close event to confirm exit and also close viewer properly.
        """
        for b in self.mBoxes:
            try:
                b.close()
            except:
                pass
        
        
        
        
        
        

global Viewer
Viewer = None
def imshow(title, image):
    """
    This method was created to display images without the used of OpenCV library. A conflict was raised with the NITLibrary, reserving a windowing space in Windows memory (NITLibrary 3.8.0).

    Parameters
    ----------
    title : TYPE
        DESCRIPTION.
    image : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    global Viewer
    if Viewer is None:
        app._GetNativeApp()
        Viewer = Viewer2D()
    
    Viewer.addFigure(title)
    window = Viewer.getActiveBox()
    window.resize(image.shape[0]*window.scaleFactor, image.shape[1]*window.scaleFactor)
    # window.resize(image.shape[0], image.shape[1])
    vv.imshow(image)
    app.Run()
    
    
    
def main():
    image = np.random.random((500,500))
    cv2.imshow("image", image)#, cv2.waitKey(0), cv2.destroyAllWindows()
    imshow("image", image)

    
    
        
if __name__ == "__main__":
    main()
        