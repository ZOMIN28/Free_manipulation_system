import manipulation
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap

class WorkerThread(QThread):
    finished = pyqtSignal(QPixmap, object)
    error = pyqtSignal(str) 

    def __init__(self, oriPath, refPath, algorithm, device, currentref):
        super().__init__()
        self.oriPath = oriPath
        self.refPath = refPath
        self.algorithm = algorithm
        self.device = device
        self.currentref = currentref

    def run(self):
        try:
            # 这里应该是图像操作的实际代码
            pixmap,result = manipulation.manipulate(self.oriPath, self.algorithm, self.device, 
                                             self.refPath,self.currentref)
            self.finished.emit(pixmap,result) 
        except Exception as e:
            self.error.emit(str(e)) 