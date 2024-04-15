import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QMenuBar, QAction, QLabel, QVBoxLayout, QWidget, QPushButton, QComboBox, QFileDialog, QHBoxLayout, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import dlib
import cv2
from thread import WorkerThread

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Manipulation Tool")
        self.setGeometry(100, 100, 800, 600)

        self.initUI()

    def initUI(self):
        self.menuBar = QMenuBar(self)
        self.setMenuBar(self.menuBar)
        self.algorithmMenu = QMenu("&algorithm", self)
        self.menuBar.addMenu(self.algorithmMenu)

       
        algorithms = ["StarGANv2", "HiSD", "HFGI", "SimSwap"]
        self.algorithmActions = {}
        for alg in algorithms:
            action = QAction(alg, self)
            action.triggered.connect(self.select_algorithm)
            self.algorithmMenu.addAction(action)
            self.algorithmActions[alg] = action

  
        info_layout = QHBoxLayout()
        self.currentAlgorithmLabel = QLabel("Please select an algorithm")
        self.currentAlgorithmLabel.setAlignment(Qt.AlignCenter)
        self.currentAlgorithmLabel.setStyleSheet("font-size: 16px;")
        self.currentAlgorithmLabel.setFixedHeight(30)
        info_layout.addWidget(self.currentAlgorithmLabel)

        self.currentDeviceLabel = QLabel("Selected device: CPU")
        self.currentDeviceLabel.setAlignment(Qt.AlignCenter)
        self.currentDeviceLabel.setStyleSheet("font-size: 16px;")
        self.currentDeviceLabel.setFixedHeight(30)
        info_layout.addWidget(self.currentDeviceLabel)

        self.statusLabel = QLabel("Current status: Finish") 
        info_layout.addWidget(self.statusLabel)

        self.devicemMenu = QMenu("&device", self)
        self.menuBar.addMenu(self.devicemMenu)
        device = ["CPU", "GPU"]
        self.deviceActions = {}
        for dev in device:
            action = QAction(dev, self)
            action.triggered.connect(self.select_device)
            self.devicemMenu.addAction(action)
            self.deviceActions[dev] = action


        self.imageLabels = {}
        self.imagePath = {}
        self.titles = ["Original image", "Referance image", "Manipulated image"]
        image_layout = QHBoxLayout()

        self.currentOriimage = QLabel(self.titles[0])
        self.currentOriimage.setAlignment(Qt.AlignCenter)
        self.currentOriimage.setFixedSize(256, 256)
        self.currentOriimage.setStyleSheet("QLabel { background-color : white; border: 1px solid black; }")
        self.currentOriimage.mousePressEvent = lambda event, t=self.titles[0]: self.open_image(t)
        image_layout.addWidget(self.currentOriimage)
        self.imageLabels[self.titles[0]] = self.currentOriimage

        self.currentRefimage = QLabel(self.titles[1])
        self.currentRefimage.setAlignment(Qt.AlignCenter)
        self.currentRefimage.setFixedSize(256, 256)
        self.currentRefimage.setStyleSheet("QLabel { background-color : white; border: 1px solid black; }")
        self.currentRefimage.mousePressEvent = lambda event, t=self.titles[1]: self.open_image(t)
        image_layout.addWidget(self.currentRefimage)
        self.imageLabels[self.titles[1]] = self.currentRefimage

        self.currentManimage = QLabel(self.titles[2])
        self.currentManimage.setAlignment(Qt.AlignCenter)
        self.currentManimage.setFixedSize(256, 256)
        self.currentManimage.setStyleSheet("QLabel { background-color : white; border: 1px solid black; }")
        image_layout.addWidget(self.currentManimage)
        self.imageLabels[self.titles[2]] = self.currentManimage

    
        self.featureCombo = QComboBox()
        self.featureCombo.setFixedWidth(256)
        image_layout.addWidget(self.featureCombo)

    
        self.doButton = QPushButton("Do it")
        self.doButton.clicked.connect(self.manipulate_image)
        image_layout.addWidget(self.doButton)

        operate_layout = QHBoxLayout()
        self.clearButton = QPushButton("Clear all")
        self.clearButton.clicked.connect(self.clear_all)
        operate_layout.addWidget(self.clearButton)
        self.saveButton = QPushButton("Save result")
        self.saveButton.clicked.connect(self.save_image)
        operate_layout.addWidget(self.saveButton)

       
        layout = QVBoxLayout()
        layout.addLayout(info_layout)
        layout.addLayout(image_layout)
        layout.addLayout(operate_layout)

    
        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)


        self.currentAlgorithm = None
        self.currentDevice = "CPU"


    def select_algorithm(self):
        action = self.sender()
        self.currentAlgorithm = action.text()
        self.currentAlgorithmLabel.setText(f"Selected algorithm: {self.currentAlgorithm}")
        self.update_feature_combo()
        self.update_ref_combo()

    def select_device(self):
        action = self.sender()
        self.currentDevice = action.text()
        self.currentDeviceLabel.setText(f"Selected device: {self.currentDevice}")
        self.update_feature_combo()

    def update_feature_combo(self):
        self.featureCombo.clear()
        features = {
            "HFGI": ["age", "smile", "eyes", "beard", "lip"],
            "HiSD": ["hair color", "glasses"],
            "StarGANv2": ["male","female"]
        }.get(self.currentAlgorithm, [])
        self.featureCombo.addItems(features)

    def update_ref_combo(self):
        if self.currentAlgorithm == "HFGI":
            self.currentRefimage.setStyleSheet("QLabel { background-color : black; border: 1px solid black; }")
        else:
            self.currentRefimage.setStyleSheet("QLabel { background-color : white; border: 1px solid black; }")

    def get_current_feature_selection(self):
        selected_feature = self.featureCombo.currentText()
        return selected_feature


    def manipulate_image(self):
        if self.currentAlgorithm == None:
            QMessageBox.warning(self, "Error", "Please select the algorithm.")
            return
        if not self.imageLabels["Original image"].pixmap():
            QMessageBox.warning(self, "Error", "Please select images for the original slots.")
            return
        if self.currentAlgorithm is not "HFGI":
            if not self.imageLabels["Original image"].pixmap():
                QMessageBox.warning(self, "Error", "Please select images for the original slots.")
                return      
            
        currentref = self.get_current_feature_selection()
        if self.currentAlgorithm == "HFGI":
            self.imagePath["Referance image"] = " "

        self.thread = WorkerThread(
            self.imagePath["Original image"],
            self.imagePath["Referance image"],
            self.currentAlgorithm,
            self.currentDevice,
            currentref
        )
        self.thread.finished.connect(self.on_finish)
        self.thread.error.connect(self.on_error)
        self.thread.start()
        self.statusLabel.setText("Current status: Waiting...") 
            

    def on_finish(self, pixmap, result):
        self.imageLabels["Manipulated image"].setPixmap(pixmap)
        self.statusLabel.setText("Current status: Finish")  

    def on_error(self, message):
        QMessageBox.critical(self, "Error", message)
        self.statusLabel.setText("Current status: Finish") 

    def save_image(self):
        if self.imageLabels["Manipulated image"].pixmap():
            filepath, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "JPEG Files (*.jpg);;PNG Files (*.png)")
            if filepath:
                self.imageLabels["Manipulated image"].pixmap().save(filepath)
            QMessageBox.information(self,"Info","Saved successfully.")
        else:
            QMessageBox.warning(self, "Warning!", "There is no generated image to save, please perform image operations first.")
        

    def open_image(self, title):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if filepath:
            filepath = self.process_img(filepath,title)
            self.imagePath[title] = filepath 
            pixmap = QPixmap(filepath).scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.imageLabels[title].setPixmap(pixmap)

    def process_img(self,filepath,title):
        detector = dlib.get_frontal_face_detector()
        image = cv2.imread(filepath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) > 0:
            if image.shape[0] == 256 and image.shape[1] == 256:
                return filepath
            face = faces[0]
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            for i in range(1,100):
                try:
                    face_region = image[y-int(h/i):y+h+int(h/5+i), x-int(w/(i+10)):x+w+int(w/(i+10))]
                    cv2.imwrite('temp/'+title+'.png', face_region)
                    break
                except:
                    pass

            return 'temp/'+title+'.png'
        else:
            QMessageBox.warning(self, "Warning!", "No face detected.")
            self.clear_all()
    
    def clear_all(self):
        self.currentOriimage.clear()
        self.currentOriimage.setText(self.titles[0])
        self.currentRefimage.clear()
        self.currentRefimage.setText(self.titles[1])
        self.currentManimage.clear()
        self.currentManimage.setText(self.titles[2])


