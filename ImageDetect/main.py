from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from pydantic import BaseModel
from PIL import Image
import numpy as np
import io
import base64

app = FastAPI()

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (보안상 주의 필요)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# YOLO 모델 로드 (예: YOLOv8n)
model = YOLO("./runs/detect/train/weights/best.pt")


class DetectionResult(BaseModel):
    message: str
    image: str  # base64 인코딩된 결과 이미지


# 객체 탐지 함수
def detect_objects(image: Image):
    img_array = np.array(image)
    results = model(img_array)  # YOLO 모델을 통한 객체 탐지
    detected_img = results.plot()  # 결과 이미지 (바운딩 박스 표시)
    return Image.fromarray(detected_img)


@app.post("/detect", response_model=DetectionResult)
async def detect_service(file: UploadFile = File(...)):
    try:
        # 이미지를 읽고 PIL 형식으로 변환
        image = Image.open(io.BytesIO(await file.read()))

        # RGB 변환 (필요할 경우)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 객체 탐지 수행
        result_image = detect_objects(image)

        # 결과 이미지를 base64로 인코딩
        buffered = io.BytesIO()
        result_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return DetectionResult(message="Detection completed", image=img_str)

    except Exception as e:
        return DetectionResult(message=f"Error: {str(e)}", image="")


# 서버 실행 명령: uvicorn main:app --reload
