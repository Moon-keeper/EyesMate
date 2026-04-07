import cv2
import mediapipe as mp
import numpy as np

# --- 1. 初始化 MediaPipe (新版写法) ---
# 新版 MediaPipe 使用 tasks.vision 来创建模型
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 配置选项
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'), # 这里会自动下载模型
    running_mode=VisionRunningMode.IMAGE, # 我们一帧一帧处理，所以用 IMAGE 模式
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 创建面部关键点检测器
landmarker = FaceLandmarker.create_from_options(options)

# --- 2. 打开摄像头 ---
cap = cv2.VideoCapture(0)

print("系统启动中... 请按 'q' 键退出")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("无法读取摄像头画面")
        break

    # 获取画面尺寸
    h, w, _ = image.shape
    
    # MediaPipe 需要 RGB 格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 将 numpy 数组转换为 MediaPipe 需要的格式 (MPImage)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # --- 3. 进行检测 ---
    # 注意：新版 detect 返回的结果结构略有不同
    detection_result = landmarker.detect(mp_image)

    # 初始化中心点变量
    head_center_x, head_center_y = 0, 0
    has_face = False

    # --- 4. 处理结果 ---
    if detection_result.face_landmarks:
        has_face = True
        # 获取第一张脸的关键点列表
        face_landmarks = detection_result.face_landmarks[0]
        
        # 收集所有关键点的坐标
        x_coords = []
        y_coords = []
        
        for landmark in face_landmarks:
            # landmark.x 和 landmark.y 是 0-1 之间的相对值，需要乘以宽高
            x_coords.append(landmark.x * w)
            y_coords.append(landmark.y * h)
            
            # 【可视化】在屏幕上画出每一个关键点（因为新版不自动连线了）
            # 我们只画绿色的点
            cv2.circle(image, (int(landmark.x * w), int(landmark.y * h)), 1, (0, 255, 0), -1)

        # 计算中心点
        head_center_x = int(np.mean(x_coords))
        head_center_y = int(np.mean(y_coords))

        # 画一个大的红点表示头部中心
        cv2.circle(image, (head_center_x, head_center_y), 8, (0, 0, 255), -1)
        
        # 显示坐标
        cv2.putText(image, f"Target: {head_center_x}, {head_center_y}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示状态
        cv2.putText(image, "Status: Tracking", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(image, "Status: No Face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # --- 5. 显示画面 ---
    cv2.imshow('EyesMate - begin', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# 释放资源
landmarker.close()
cap.release()
cv2.destroyAllWindows()