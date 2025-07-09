camera:
  source: 0  # 0 for webcam, or RTSP URL
  width: 1280
  height: 720

models:
  weapon: "models/yolov9_weapons.pt"
  violence: "models/stgcn_fighting.onnx"
  fire: "models/efficientnet_b7.pth"

alerts:
  sound: "alarm.wav"
  sms_notifications: false