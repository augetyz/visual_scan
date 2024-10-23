from flask import Flask, Response, render_template, request
import cv2
import numpy as np
import threading
import os
from periphery import Serial  # 导入串口通信库
import time  # 导入时间库
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# 初始化摄像头
camera = cv2.VideoCapture(0)  # 使用默认摄像头
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# camera.set(cv2.CAP_PROP_FPS, 30)  # 设置为 30 FPS
camera2 = cv2.VideoCapture(2)  # 使用第二个摄像头
camera2.set(cv2.CAP_PROP_FRAME_WIDTH,  720)
camera2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# camera2.set(cv2.CAP_PROP_FPS, 30)  # 设置为 30 FPS
ser=None
def serial_open():
    global ser
    available_ports = ["/dev/ttyACM0", "/dev/ttyACM1"]  # 可用的端口列表
    while True:
        for port in available_ports:
            if os.path.exists(port):  # 检测设备是否存在
                try:
                    ser = Serial(port, 115200)
                    print(f"Serial opened on {port}")
                    return  # 成功打开后退出函数
                except Exception as e:
                    print(f"Failed to open {port}: {e}")
        print("No valid serial port found. Retrying...")
        time.sleep(0.1)  # 延迟后重试
# 定义颜色阈值（HSV格式）
thresholds = {
    'red': {'lower': np.array([0, 20, 20]), 'upper': np.array([40, 255, 234])},
    'green': {'lower': np.array([40,40,40]), 'upper': np.array([80,255,255])},
    'blue': {'lower': np.array([34, 90, 70]), 'upper': np.array([100, 255, 255])}
}

# 定义色块检测的阈值
block_thresholds = {
    'red': {'lower': np.array([0, 100, 100]), 'upper': np.array([10, 255, 255])},
    'green': {'lower': np.array([40, 100, 100]), 'upper': np.array([80, 255, 255])},
    'blue': {'lower': np.array([100, 150, 0]), 'upper': np.array([140, 255, 255])}
}

# 定义腐蚀内核
kernel = np.ones((1, 1), np.uint8)

# 存储视频帧的全局变量
output_frame = None
lock = threading.Lock()  # 线程锁

task = 'block'
data_queue = Queue()

def process_camera_feed():
    min_radius = 10
    max_radius = 200

    min_area = 500  # 最小面积阈值
    max_area = 50000  # 最大面积阈值

    global output_frame
    global thresholds
    def send_data_to_queue(x, y, color, frame_type):
        # 将 x 和 y 转换为 Python 原生 int 类型，再调用 to_bytes
        x_bytes = int(x).to_bytes(2, byteorder='big')
        y_bytes = int(y).to_bytes(2, byteorder='big')

        # 颜色转换为单个字节
        color_byte = bytes([color])

        # 构建数据内容
        data_content = x_bytes + y_bytes + color_byte
        data_length = len(data_content)

        # 构建最终帧
        frame = bytearray([0xAA, frame_type, data_length]) + data_content + bytearray([0xBB])

        # 将帧放入队列
        data_queue.put(frame)

    # # 记录开始时间
    # start_time = time.time()
    # frame_count = 0
    while True:
        success, frame = camera.read()  # 从摄像头读取帧
        if not success:
            break
        # # 增加帧计数
        # frame_count += 1
        # 创建彩色图像的副本用于标注
        annotated_frame = frame.copy()
        mask_red = None
        mask_green = None
        mask_blue = None
        # 转换为HSV格式
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if task == 'circle':
            # 使用动态阈值创建掩膜（色环检测）
            mask_red = cv2.inRange(hsv, thresholds['red']['lower'], thresholds['red']['upper'])
            mask_red = cv2.erode(mask_red, kernel, iterations=1)

            mask_green = cv2.inRange(hsv, thresholds['green']['lower'], thresholds['green']['upper'])
            mask_green = cv2.erode(mask_green, kernel, iterations=1)

            mask_blue = cv2.inRange(hsv, thresholds['blue']['lower'], thresholds['blue']['upper'])
            mask_blue = cv2.erode(mask_blue, kernel, iterations=1)

            # 对每个颜色进行圆形检测
            circles_red = cv2.HoughCircles(mask_red, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                           param1=50, param2=80, minRadius=10, maxRadius=200)
            circles_green = cv2.HoughCircles(mask_green, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                             param1=50, param2=80, minRadius=10, maxRadius=200)
            circles_blue = cv2.HoughCircles(mask_blue, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                            param1=50, param2=80, minRadius=10, maxRadius=200)

            # 查找最大圆形的通用函数
            def find_largest_circle(circles):
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    largest_circle = max(circles, key=lambda x: x[2])  # 根据半径进行排序
                    return tuple(largest_circle)  # 返回 (x, y, r) 作为元组
                return None

            # 标注红色最大圆形并打包坐标和半径
            largest_red_circle = find_largest_circle(circles_red)
            if largest_red_circle is not None:
                (x, y, r) = largest_red_circle
                cv2.circle(annotated_frame, (x, y), r, (0, 0, 255), 4)
                cv2.circle(annotated_frame, (x, y), 2, (0, 0, 255), 3)  # 标记中心点
                print(f"Red Circle - {(x, y, r)}")
                send_data_to_queue(x, y, 0x01, 0xDD)  # Red circle

            # 标注绿色最大圆形
            largest_green_circle = find_largest_circle(circles_green)
            if largest_green_circle is not None:
                (x, y, r) = largest_green_circle
                cv2.circle(annotated_frame, (x, y), r, (0, 255, 0), 4)
                cv2.circle(annotated_frame, (x, y), 2, (0, 255, 0), 3)  # 标记中心点
                print(f"Green Circle - {(x, y, r)}")
                send_data_to_queue(x, y, 0x02, 0xDD)  # Green circle

            # 标注蓝色最大圆形
            largest_blue_circle = find_largest_circle(circles_blue)
            if largest_blue_circle is not None:
                (x, y, r) = largest_blue_circle
                cv2.circle(annotated_frame, (x, y), r, (255, 0, 0), 4)
                cv2.circle(annotated_frame, (x, y), 2, (255, 0, 0), 3)  # 标记中心点
                print(f"Blue Circle - {(x, y, r)}")
                send_data_to_queue(x, y, 0x03, 0xDD)  # Blue circle


        elif task == 'block':

            # 使用动态阈值创建掩膜（色块检测）

            mask_red = cv2.inRange(hsv, block_thresholds['red']['lower'], block_thresholds['red']['upper'])

            mask_green = cv2.inRange(hsv, block_thresholds['green']['lower'], block_thresholds['green']['upper'])

            mask_blue = cv2.inRange(hsv, block_thresholds['blue']['lower'], block_thresholds['blue']['upper'])

            # 对每个颜色进行色块检测

            def find_largest_contour(mask, min_area, max_area):
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                largest_contour = None
                max_area_found = 0  # 局部变量来存储最大面积
                for contour in contours:
                    area = cv2.contourArea(contour)
                    # 检查最小和最大面积阈值
                    if min_area < area < max_area:
                        if area > max_area_found:  # 更新最大轮廓
                            max_area_found = area
                            largest_contour = contour
                return largest_contour

            min_area = 500  # 定义最小面积阈值
            # 标注最大红色色块
            largest_red_contour = find_largest_contour(mask_red, min_area, max_area)
            if largest_red_contour is not None:
                x, y, w, h = cv2.boundingRect(largest_red_contour)
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                center = (x + w // 2, y + h // 2)
                print(f"Red Block - Center: {center}, Width: {w}, Height: {h}")
                send_data_to_queue(center[0], center[1], 0x01, 0xEE)  # Red block
            # 标注最大绿色色块
            largest_green_contour = find_largest_contour(mask_green, min_area, max_area)
            if largest_green_contour is not None:
                x, y, w, h = cv2.boundingRect(largest_green_contour)
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                center = (x + w // 2, y + h // 2)
                print(f"Green Block - Center: {center}, Width: {w}, Height: {h}")
                send_data_to_queue(center[0], center[1], 0x02, 0xEE)  # Green block
            # 标注最大蓝色色块
            largest_blue_contour = find_largest_contour(mask_blue, min_area, max_area)
            if largest_blue_contour is not None:
                x, y, w, h = cv2.boundingRect(largest_blue_contour)
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                center = (x + w // 2, y + h // 2)
                print(f"Blue Block - Center: {center}, Width: {w}, Height: {h}")
                send_data_to_queue(center[0], center[1], 0x03, 0xEE)  # Blue block

        # # 计算并显示帧率（FPS）
        # elapsed_time = time.time() - start_time
        # if elapsed_time > 0:
        #     fps = frame_count / elapsed_time
        #     # 将FPS绘制在图像上
        #     cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #                 (255, 255, 255), 2)
        # 更新全局帧变量（线程安全）
        with lock:
            output_frame = {
                'color': annotated_frame,
                'red': mask_red,
                'green': mask_green,
                'blue': mask_blue
            }
# 定义全局变量用于控制二维码扫描线程
qr_code_scanning_enabled = True
def process_qr_code():
    global qr_code_scanning_enabled  # 引用全局变量
    qr_detector = cv2.QRCodeDetector()  # 创建 QRCodeDetector 实例
    def send_qr_data_to_queue(data):
        # Assuming data is a string of 6 characters
        data_content = data.encode('utf-8')[:3]+data.encode('utf-8')[4:7]  
        data_length = len(data_content)
        frame = bytearray([0xAA, 0xCC, data_length]) + data_content + bytearray([0xBB])
        data_queue.put(frame)

    while True:
        if not qr_code_scanning_enabled:  # 检查是否允许扫描
            time.sleep(0.1)
            continue  # 如果不允许，继续下一次循环

        success, frame = camera2.read()  # 从第二个摄像头读取帧
        if not success:
            continue  # 如果读取失败，继续下一次循环

        if frame is None or frame.size == 0:
            print("Error: Received empty frame.")
            continue
        try:
            # 使用 detectAndDecode 方法检测 QR 码
            data, bbox, _ = qr_detector.detectAndDecode(frame)

            if bbox is not None and data:
                # 确保 bbox 是整数并绘制多边形
                bbox = bbox.astype(int)
                cv2.polylines(frame, [bbox], isClosed=True, color=(0, 255, 0), thickness=2)
                print(f"Detected QR code: {data}")
                send_qr_data_to_queue(data)
        except cv2.error as e:
            print(f"OpenCV error during QR code detection: {e}")

        # 更新全局帧变量（线程安全）
        with lock:
            if output_frame is not None:  # 检查 output_frame 是否已初始化
                output_frame['qrcode'] = frame


def process_serial_communication():
    global task
    global qr_code_scanning_enabled
    wait_time=0
    while True:
        try:
            # 检查是否有数据等待读取
            num_data = ser.input_waiting()
            if num_data > 0:
                # 读取串口数据
                line = ser.read(num_data, 0)
                print(f"Received: {line}")
                for byte in line:
                    # 根据接收到的字节修改task和qr_code_scanning_enabled
                    if byte == ord('1'):
                        task = 'block'
                        print('block')
                    elif byte == ord('2'):
                        task = 'circle'
                        print('circle')
                    elif byte == ord('3'):
                        qr_code_scanning_enabled = True  # 启用二维码扫描
                        print('qr_code_scanning_enabled')
                    elif byte == ord('4'):
                        qr_code_scanning_enabled = False  # 禁用二维码扫描
                        print('qr_code_scanning_disabled')
        except Exception as e:
            print("serial error")
            serial_open()
        # 从队列中获取数据并通过串口发送
        if not data_queue.empty():
            data_to_send = data_queue.get()
            ser.write(data_to_send)
            print(f"Sent to serial: {data_to_send.hex()}")
        time.sleep(0.1)  # 暂停以降低CPU使用率
        

def generate_frames(color):
    global output_frame
    while True:
        with lock:
            if output_frame is None or color not in output_frame:
                continue
            # 根据请求的颜色编码图像
            try:
                ret, buffer = cv2.imencode('.jpg', output_frame[color])
                frame = buffer.tobytes()
            except KeyError:
                continue  # 如果键不存在，继续下一次循环

        # 返回单帧
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/color')
def video_feed_color():
    return Response(generate_frames('color'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/red')
def video_feed_red():
    return Response(generate_frames('red'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/green')
def video_feed_green():
    return Response(generate_frames('green'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/blue')
def video_feed_blue():
    return Response(generate_frames('blue'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video_feed/qrcode')
def video_feed_qrcode():
    return Response(generate_frames('qrcode'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_thresholds', methods=['POST'])
def update_thresholds():
    global thresholds
    thresholds['red']['lower'] = np.array([int(request.form.get('red_lower_h')),
                                           int(request.form.get('red_lower_s')),
                                           int(request.form.get('red_lower_v'))])
    thresholds['red']['upper'] = np.array([int(request.form.get('red_upper_h')),
                                           int(request.form.get('red_upper_s')),
                                           int(request.form.get('red_upper_v'))])
    thresholds['green']['lower'] = np.array([int(request.form.get('green_lower_h')),
                                             int(request.form.get('green_lower_s')),
                                             int(request.form.get('green_lower_v'))])
    thresholds['green']['upper'] = np.array([int(request.form.get('green_upper_h')),
                                             int(request.form.get('green_upper_s')),
                                             int(request.form.get('green_upper_v'))])
    thresholds['blue']['lower'] = np.array([int(request.form.get('blue_lower_h')),
                                            int(request.form.get('blue_lower_s')),
                                            int(request.form.get('blue_lower_v'))])
    thresholds['blue']['upper'] = np.array([int(request.form.get('blue_upper_h')),
                                            int(request.form.get('blue_upper_s')),
                                            int(request.form.get('blue_upper_v'))])
    return 'Thresholds updated successfully!'


# 初始化一个ThreadPoolExecutor，工作线程数设置为CPU核心数
executor = ThreadPoolExecutor(max_workers=8)  # 根据需要调整工作线程数

if __name__ == '__main__':
    serial_open()
    # 使用executor并行运行摄像头和二维码处理
    executor.submit(process_camera_feed)
    executor.submit(process_qr_code)
    executor.submit(process_serial_communication)
    
    #启动Flask服务，访问 http://<服务器IP>:5000 查看识别结果
    app.run(host='0.0.0.0', port=5000)
