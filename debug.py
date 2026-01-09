import cv2

cap = cv2.VideoCapture(1)

# Đọc 1 frame
ret, frame = cap.read()

if ret:
    h, w = frame.shape[:2]
    print("=" * 50)
    print(f"🎥 WEBCAM CỦA BẠN: {w} x {h} PIXELS")
    print("=" * 50)
    
    # Kiểm tra các độ phân giải hỗ trợ
    print("\nCác độ phân giải webcam hỗ trợ:")
    resolutions = [
        (640, 480),   # VGA
        (800, 600),   # SVGA  
        (1024, 768),  # XGA
        (1280, 720),  # HD
        (1920, 1080)  # Full HD
    ]
    
    for res_w, res_h in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_h)
        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        if actual_w == res_w and actual_h == res_h:
            print(f"✓ {res_w}x{res_h}")
        else:
            print(f"✗ {res_w}x{res_h} (thực tế: {actual_w}x{actual_h})")

cap.release()