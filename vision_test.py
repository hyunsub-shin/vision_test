import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic
import datetime
import os

# UI 파일 로드
form_class = uic.loadUiType("vision_test.ui")[0]

class SMTInspectionApp(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.reference_image = None
        self.current_image = None
        
        # 카메라 선택 콤보박스 설정
        self.camera_list = self.get_available_cameras()
        self.camera_comboBox.addItems([f"카메라 {i}" for i in range(len(self.camera_list))])
        self.camera_comboBox.currentIndexChanged.connect(self.change_camera)
        
        # 기본 카메라 연결
        self.current_camera_index = 0
        self.camera = cv2.VideoCapture(self.current_camera_index)
        
        # 카메라 연결 확인
        if not self.camera.isOpened():
            QMessageBox.critical(self, "오류", "카메라를 찾을 수 없습니다.")
            sys.exit()
            
        self.counter = 1
        self.ng_threshold = 15  # 기본값 조정
        # self.threshold_slider.setMinimum(1)
        # self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(self.ng_threshold)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        
        # 초기값을 라벨에 표시
        self.threshold_label.setText(f"임계값: {self.ng_threshold}")
          
        # 버튼 연결
        self.ref_button.clicked.connect(self.set_reference)
        
        # 카메라 상태 변수 추가
        self.camera_running = False
        
        # Start/Stop 버튼 연결
        self.start_stop_button.clicked.connect(self.toggle_camera)
        
        # 타이머 설정 - 카메라 프레임 업데이트
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000)  # 1000ms 간격으로 프레임 업데이트

        # # 카메라 해상도 설정
        # self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 너비
        # self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) # 높이
        # # 프레임레이트 설정
        # self.camera.set(cv2.CAP_PROP_FPS, 30)           # 30fps

        # ROI 관련 변수 초기화
        self.roi = None
        self.drawing = False
        self.roi_start = None
        self.roi_selection_mode = False
        
        # ROI 버튼 이벤트 연결 (버튼은 UI 파일에서 정의)
        self.roi_button.clicked.connect(self.start_roi_selection)
        
        # 카메라 레이블에 마우스 이벤트 연결
        self.camera_label.mousePressEvent = self.mouse_press
        self.camera_label.mouseReleaseEvent = self.mouse_release
        self.camera_label.mouseMoveEvent = self.mouse_move

        # 메시지 표시 상태를 저장하는 변수 추가
        self.roi_warning_shown = False

    def get_available_cameras(self):
        """사용 가능한 카메라 목록 반환"""
        available_cameras = []
        # 검색 범위를 3개로 제한 (일반적으로 0, 1, 2만 사용)
        for i in range(3):  
            try:
                # 먼저 기본 백엔드로 시도
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:  # 프레임이 실제로 존재하는지 확인
                        available_cameras.append(i)
                cap.release()
            except Exception as e:
                print(f"카메라 {i} 검색 중 오류: {str(e)}")
                continue
                
        if not available_cameras:  # 사용 가능한 카메라가 없으면
            print("사용 가능한 카메라가 없습니다. 기본 카메라(0)를 사용합니다.")
            return [0]
            
        return available_cameras

    def change_camera(self, index):
        """카메라 변경"""
        try:
            if self.camera_running:
                self.toggle_camera()  # 현재 카메라 중지
            
            self.current_camera_index = self.camera_list[index]
            
            # 카메라 초기화 전에 이전 인스턴스 해제
            if hasattr(self, 'camera') and self.camera is not None:
                self.camera.release()
                
            # 기본 백엔드로 카메라 초기화
            self.camera = cv2.VideoCapture(self.current_camera_index)
            
            if not self.camera.isOpened():
                QMessageBox.warning(self, "경고", f"카메라 {self.current_camera_index}를 열 수 없습니다.")
                return
                
            # # 카메라가 성공적으로 열렸을 때만 설정 적용
            # self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            # self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            if self.camera_running:
                self.toggle_camera()  # 새 카메라 시작
        except Exception as e:
            QMessageBox.warning(self, "오류", f"카메라 변경 중 오류가 발생했습니다: {str(e)}")

    def toggle_camera(self):
        if not self.camera_running:
            # 카메라 시작
            self.camera_running = True
            self.start_stop_button.setText("Stop")
            self.timer.start(1000)  # 1000ms 간격으로 프레임 업데이트 (약 1fps)
        else:
            # 카메라 정지
            self.camera_running = False
            self.start_stop_button.setText("Start")
            self.timer.stop()
            
            # 모든 상태 초기화
            self.camera_label.clear()
            self.camera_label.setText("Camera Stopped")
            self.reference_image = None
            self.reference_label.clear()
            self.reference_label.setText("기준 이미지")
            self.result_label.clear()
            self.result_label.setText("검사결과")
            self.result_label.setStyleSheet("background-color: #f0f0f0;")
            self.roi = None
            self.drawing = False
            self.roi_start = None
            self.roi_selection_mode = False
            self.roi_warning_shown = False  # 경고 상태도 초기화

    def update_frame(self):
        if not self.camera_running:
            return
            
        ret, frame = self.camera.read()
        if ret:
            self.current_image = frame
            display_frame = frame.copy()
            
            # ROI가 설정되어 있다면 항상 표시
            if self.roi is not None:
                x1, y1, x2, y2 = self.roi
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            self.display_image(display_frame)
            if self.reference_image is not None:
                self.inspect_image()
        else:
            # 프레임 읽기 실패 시 재시도
            self.camera.release()
            self.camera = cv2.VideoCapture(self.current_camera_index)  # 카메라 재연결 시도
            if not self.camera.isOpened():
                QMessageBox.critical(self, "오류", "카메라 연결이 끊어졌습니다.")
                self.camera_running = False
                self.start_stop_button.setText("Start")
                self.timer.stop()

    def display_image(self, image):
        h, w, ch = image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(image.data, w, h, bytes_per_line, QImage.Format_BGR888)
        scaled_image = convert_to_qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        self.camera_label.setPixmap(QPixmap.fromImage(scaled_image))

    def set_reference(self):
        if not self.camera_running:
            QMessageBox.warning(self, "경고", "카메라를 먼저 시작해주세요.")
            return
            
        if self.roi is None and not self.roi_warning_shown:
            self.roi_warning_shown = True
            QMessageBox.warning(self, "경고", "ROI를 먼저 선택해주세요.")
            return
            
        if self.current_image is not None and self.roi is not None:
            self.reference_image = self.current_image.copy()
            self.display_reference_image(self.reference_image)
            self.roi_warning_shown = False  # 참조 이미지 설정 성공 시 경고 상태 초기화

    def display_reference_image(self, image):
        display_ref = image.copy()
        
        # 참조 이미지에도 ROI 표시
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            cv2.rectangle(display_ref, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        h, w, ch = display_ref.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(display_ref.data, w, h, bytes_per_line, QImage.Format_BGR888)
        scaled_image = convert_to_qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        self.reference_label.setPixmap(QPixmap.fromImage(scaled_image))

    def update_threshold(self, value):
        self.ng_threshold = value
        self.threshold_label.setText(f"임계값: {value}")

    def start_roi_selection(self):
        if not self.roi_selection_mode:  # ROI 선택 모드가 아닐 때만 실행
            self.roi_selection_mode = True
            self.drawing = False
            self.roi = None
            QMessageBox.information(self, "안내", "카메라 화면에서 관심영역을 드래그하여 선택하세요.")

    def mouse_press(self, event):
        if self.current_image is not None:
            self.drawing = True
            # QLabel의 실제 표시 크기 계산
            label_width = self.camera_label.width()
            label_height = self.camera_label.height()
            
            # 이미지 크기
            img_height, img_width = self.current_image.shape[:2]
            
            # 이미지의 실제 표시 크기 계산 (종횡비 유지)
            if img_width / label_width > img_height / label_height:
                display_width = label_width
                display_height = int(img_height * label_width / img_width)
            else:
                display_height = label_height
                display_width = int(img_width * label_height / img_height)
            
            # 이미지가 중앙에 표시되므로 오프셋 계산
            x_offset = (label_width - display_width) // 2
            y_offset = (label_height - display_height) // 2
            
            # 마우스 좌표를 실제 이미지 좌표로 변환
            x = event.pos().x() - x_offset
            y = event.pos().y() - y_offset
            
            if x >= 0 and y >= 0 and x < display_width and y < display_height:
                scale_x = img_width / display_width
                scale_y = img_height / display_height
                self.roi_start = (int(x * scale_x), int(y * scale_y))

    def mouse_move(self, event):
        if self.drawing and self.current_image is not None:
            temp_image = self.current_image.copy()
            
            # QLabel의 실제 표시 크기 계산
            label_width = self.camera_label.width()
            label_height = self.camera_label.height()
            
            # 이미지 크기
            img_height, img_width = self.current_image.shape[:2]
            
            # 이미지의 실제 표시 크기 계산 (종횡비 유지)
            if img_width / label_width > img_height / label_height:
                display_width = label_width
                display_height = int(img_height * label_width / img_width)
            else:
                display_height = label_height
                display_width = int(img_width * label_height / img_height)
            
            # 이미지가 중앙에 표시되므로 오프셋 계산
            x_offset = (label_width - display_width) // 2
            y_offset = (label_height - display_height) // 2
            
            # 마우스 좌표를 실제 이미지 좌표로 변환
            x = event.pos().x() - x_offset
            y = event.pos().y() - y_offset
            
            if x >= 0 and y >= 0 and x < display_width and y < display_height:
                scale_x = img_width / display_width
                scale_y = img_height / display_height
                current_pos = (int(x * scale_x), int(y * scale_y))
                cv2.rectangle(temp_image, self.roi_start, current_pos, (0, 255, 0), 2)
                self.display_image(temp_image)

    def mouse_release(self, event):
        if self.drawing and self.current_image is not None:
            # QLabel의 실제 표시 크기 계산
            label_width = self.camera_label.width()
            label_height = self.camera_label.height()
            
            # 이미지 크기
            img_height, img_width = self.current_image.shape[:2]
            
            # 이미지의 실제 표시 크기 계산 (종횡비 유지)
            if img_width / label_width > img_height / label_height:
                display_width = label_width
                display_height = int(img_height * label_width / img_width)
            else:
                display_height = label_height
                display_width = int(img_width * label_height / img_height)
            
            # 이미지가 중앙에 표시되므로 오프셋 계산
            x_offset = (label_width - display_width) // 2
            y_offset = (label_height - display_height) // 2
            
            # 마우스 좌표를 실제 이미지 좌표로 변환
            x = event.pos().x() - x_offset
            y = event.pos().y() - y_offset
            
            if x >= 0 and y >= 0 and x < display_width and y < display_height:
                scale_x = img_width / display_width
                scale_y = img_height / display_height
                end_point = (int(x * scale_x), int(y * scale_y))
                
                # ROI 좌표 설정
                x1, y1 = min(self.roi_start[0], end_point[0]), min(self.roi_start[1], end_point[1])
                x2, y2 = max(self.roi_start[0], end_point[0]), max(self.roi_start[1], end_point[1])
                self.roi = (x1, y1, x2, y2)
                self.roi_selection_mode = False
                QMessageBox.information(self, "안내", "ROI가 설정되었습니다.")
            
            self.drawing = False

    def inspect_image(self):
        if self.roi is None:
            QMessageBox.warning(self, "경고", "ROI를 먼저 선택해주세요.")
            return

        # ROI 영역 추출
        x1, y1, x2, y2 = self.roi
        current_roi = self.current_image[y1:y2, x1:x2]
        reference_roi = self.reference_image[y1:y2, x1:x2]

        # 1. 그레이스케일 변환
        current_gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
        reference_gray = cv2.cvtColor(reference_roi, cv2.COLOR_BGR2GRAY)
        
        # 2. 노이즈 제거를 위한 블러링
        current_gray = cv2.GaussianBlur(current_gray, (5,5), 0)
        reference_gray = cv2.GaussianBlur(reference_gray, (5,5), 0)
        
        # 3. 차이 계산
        diff = cv2.absdiff(current_gray, reference_gray)
        
        # 4. 임계값 적용 (슬라이더 값 사용)
        _, thresh = cv2.threshold(diff, self.ng_threshold, 255, cv2.THRESH_BINARY)
        
        # 5. 모폴로지 연산으로 노이즈 제거
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 6. 컨투어 찾기
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        has_significant_diff = False
        result_image = self.current_image.copy()
        
        # ROI 영역 표시
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 최소 면적 임계값 조정
        min_area = 80  # 노이즈 제거를 위해 증가
        total_diff_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                total_diff_area += area
                cx, cy, cw, ch = cv2.boundingRect(contour)
                # ROI 내부의 좌표를 전체 이미지 좌표로 변환
                abs_x = x1 + cx
                abs_y = y1 + cy
                cv2.rectangle(result_image, (abs_x, abs_y), (abs_x+cw, abs_y+ch), (0, 0, 255), 2)
                
                # 차이점 강조
                roi = result_image[abs_y:abs_y+ch, abs_x:abs_x+cw]
                red_overlay = np.zeros_like(roi)
                red_overlay[:, :] = [0, 0, 255]
                cv2.addWeighted(red_overlay, 0.3, roi, 0.7, 0, roi)
        
        # ROI 영역 대비 차이 영역의 비율 계산
        roi_area = (x2 - x1) * (y2 - y1)
        diff_ratio = (total_diff_area / roi_area) * 100
        
        # 차이 비율이 임계값을 넘을 때만 NG 처리
        if diff_ratio > 5:  # 5%이상 차이날 때 NG
            has_significant_diff = True

        if has_significant_diff:
            self.result_label.setText("NG")
            self.result_label.setStyleSheet("background-color: #ff5252; color: white; font-size: 24px; font-weight: bold; padding: 10px; border-radius: 5px;")
            
            # NG 이미지 저장
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if not os.path.exists("NG_images"):
                os.makedirs("NG_images")
            
            comparison = np.hstack((self.reference_image, result_image))
            filename = f"NG_{self.counter}_{timestamp}.jpg"
            cv2.imwrite(f"NG_images/{filename}", comparison)
            self.counter += 1
            
            self.display_image(result_image)
        else:
            self.result_label.setText("PASS")
            self.result_label.setStyleSheet("background-color: #4caf50; color: white; font-size: 24px; font-weight: bold; padding: 10px; border-radius: 5px;")

    def closeEvent(self, event):
        # 프로그램 종료 시 카메라 해제
        self.camera.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SMTInspectionApp()
    ex.show()
    sys.exit(app.exec_())
