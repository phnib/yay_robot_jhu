import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QSpinBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
import numpy as np
import argparse
from low_level_policy import LowLevelPolicy
# from dvrk_inference_class import DvrkInference

class RobotInferenceThread(QThread):
    finished = pyqtSignal()
    update_images = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)

    def __init__(self, robot):
        super().__init__()
        self.robot = robot
        self.running = False

    def run(self):
        self.running = True
        self.robot.num_inferences = 1
        while self.running:
            self.robot.run_inference()
            left_img, lw_img, rw_img = self.robot.update_img()
            self.update_images.emit(left_img, lw_img, rw_img)
        self.finished.emit()

    def stop(self):
        self.running = False

class SurgicalRobotUI(QWidget):
    def __init__(self, robot):
        super().__init__()
        self.robot = robot
        self.initUI()
        self.inference_thread = None
        self.timer = QTimer()
        self.image_update_timer = QTimer()
        self.timer.timeout.connect(self.update_images)
        self.image_update_timer.timeout.connect(self.fetch_and_display_images)

        # Start a timer to fetch images continuously
        self.image_update_timer.start(100)  # Update every 100 ms

    def initUI(self):
        self.setWindowTitle('Surgical Robot Interface')
        self.setGeometry(100, 100, 1500, 800)

        # Layouts
        main_layout = QVBoxLayout()
        image_layout = QHBoxLayout()
        control_layout = QVBoxLayout()

        # Images
        self.left_img_label = QLabel()
        self.lw_img_label = QLabel()
        self.rw_img_label = QLabel()
        
        # Add images to image layout
        image_layout.addWidget(self.lw_img_label)
        image_layout.addWidget(self.left_img_label)
        image_layout.addWidget(self.rw_img_label)

        # Action Execution Horizon
        self.horizon_label = QLabel('Action Execution Horizon:')
        self.horizon_spinbox = QSpinBox()
        self.horizon_spinbox.setValue(self.robot.action_execution_horizon)
        self.horizon_spinbox.setMinimum(1)
        self.horizon_spinbox.setMaximum(100)

        # Command Selection
        self.command_label = QLabel('Select Command:')
        self.command_combobox = QComboBox()
        self.command_combobox.addItems(self.robot.commands)

        # Start/Stop Button
        self.start_button = QPushButton('Start Inference')
        self.start_button.clicked.connect(self.toggle_inference)

        # Add controls to control layout
        control_layout.addWidget(self.horizon_label)
        control_layout.addWidget(self.horizon_spinbox)
        control_layout.addWidget(self.command_label)
        control_layout.addWidget(self.command_combobox)
        control_layout.addWidget(self.start_button)

        # Add image layout and control layout to main layout
        main_layout.addLayout(image_layout)
        main_layout.addLayout(control_layout)

        self.setLayout(main_layout)

    def toggle_inference(self):
        if self.inference_thread is None:
            # Start inference
            self.robot.action_execution_horizon = self.horizon_spinbox.value()
            self.robot.command = self.command_combobox.currentText()
            self.inference_thread = RobotInferenceThread(self.robot)
            self.inference_thread.finished.connect(self.inference_finished)
            self.inference_thread.update_images.connect(self.update_images)
            self.inference_thread.start()
            self.start_button.setText('Stop Inference')
            self.timer.start(100)
        else:
            # Stop inference
            self.inference_thread.stop()
            self.inference_thread = None
            self.start_button.setText('Start Inference')
            self.timer.stop()

    def inference_finished(self):
        self.inference_thread = None
        self.start_button.setText('Start Inference')

    def fetch_and_display_images(self):
        left_img, lw_img, rw_img = self.robot.update_img()
        self.update_images(left_img, lw_img, rw_img)

    def update_images(self, left_img=None, lw_img=None, rw_img=None):
        if left_img is not None and lw_img is not None and rw_img is not None:
            # print("left_img shape:", left_img.shape, "left_img dtype:", left_img.dtype)
            # print("lw_img shape:", lw_img.shape, "lw_img dtype:", lw_img.dtype)
            # print("rw_img shape:", rw_img.shape, "rw_img dtype:", rw_img.dtype)
            
            self.display_image(self.left_img_label, left_img)
            self.display_image(self.lw_img_label, lw_img)
            self.display_image(self.rw_img_label, rw_img)

    def display_image(self, label, img_data):
        img_data = (img_data).astype(np.uint8)
        q_img = QImage(img_data.data, img_data.shape[1], img_data.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, help='specify ckpt file path', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--use_language', action='store_true')
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    args = parser.parse_args()

    # robot = DvrkInference(
    #     ckpt_dir=args.ckpt_dir, 
    #     policy_class=args.policy_class, 
    #     task_name=args.task_name, 
    #     seed=args.seed, 
    #     use_language=args.use_language, 
    #     num_epochs=args.num_epochs
    # )

    robot = LowLevelPolicy(args)
    app = QApplication(sys.argv)
    ex = SurgicalRobotUI(robot)
    ex.show()
    sys.exit(app.exec_())
