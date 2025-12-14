from google import genai
import asyncio 
from google.genai import types 
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import uuid 
from pathlib import Path
import hashlib
import time
import numpy as np
import cv2

from PIL import Image, ImageChops, ImageEnhance

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Gemini File Search POC")

# Template engine
templates = Jinja2Templates(directory="template")

# Mount static
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# SESSION HANDLING
# ---------------------------

SESSION_TIMEOUT = 15 * 60   # 15 phút

# Store all active sessions in RAM
active_sessions = {}  
# Structure:
# active_sessions[username] = {
#     "hashes": {},
#     "context": [],
#     "last_active": <timestamp>
# }

def update_last_active(username):
    if username in active_sessions:
        active_sessions[username]["last_active"] = time.time()

def is_session_expired(username):
    if username not in active_sessions:
        return True
    last_active = active_sessions[username]["last_active"]
    return (time.time() - last_active) > SESSION_TIMEOUT


# ---------------------------
# GEMINI CLIENT + STORE
# ---------------------------

client = genai.Client(api_key="AIzaSyC1vj7Cg-YeABOyIO9WO16DHUlBG1k0c2E")
store_name = "Proof-of-Content"

def init_store():
    global store
    stores = client.file_search_stores.list()
    for s in stores:
        if s.display_name == store_name:
            store = s
            return
    store = client.file_search_stores.create(config={"display_name": store_name})

init_store()


def file_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


# ---------------------------
# LOGIN SYSTEM
# ---------------------------

USERS = {
    "admin": "123456",
    "test": "test123",
}

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def login(request: Request):
    form = await request.form()
    username = form.get("username")
    password = form.get("password")

    if username not in USERS or USERS[username] != password:
        return HTMLResponse("Incorrect username / password!")

    # Create fresh session
    active_sessions[username] = {
        "hashes": {},
        "context": [],
        "last_active": time.time()
    }

    response = RedirectResponse("/home", status_code=302)
    response.set_cookie("user", username)
    return response


@app.get("/logout")
async def logout(request: Request):
    user = request.cookies.get("user")

    if user in active_sessions:
        active_sessions.pop(user, None)

    response = RedirectResponse("/", status_code=302)
    response.delete_cookie("user")
    return response


@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    user = request.cookies.get("user")

    if not user:  
        return RedirectResponse("/")

    if is_session_expired(user):
        active_sessions.pop(user, None)
        return RedirectResponse("/")

    update_last_active(user)

    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ---------------------------
# UPLOAD ENDPOINT
# ---------------------------

@app.post("/upload")
async def upload_file(request: Request, files: list[UploadFile] = File(...)):
    user = request.cookies.get("user")
    if not user:
        raise HTTPException(401, "Please login!")

    if is_session_expired(user):
        active_sessions.pop(user, None)
        raise HTTPException(401, "Expired session!")

    update_last_active(user)

    session_data = active_sessions[user]
    imported_hashes = session_data["hashes"]

    result = []

    for file in files:
        ext = Path(file.filename).suffix
        safe_name = f"{uuid.uuid4()}{ext}"
        file_path = UPLOAD_DIR / safe_name

        with open(file_path, "wb") as f:
            f.write(await file.read())

        sha = file_hash(file_path)

        if sha in imported_hashes:
            result.append(f"[{user}] File {file.filename} had existed")
            continue

        file_uploaded = client.files.upload(file=str(file_path))

        operation = client.file_search_stores.import_file(
            file_search_store_name=store.name,
            file_name=file_uploaded.name
        )

        while not operation.done:
            await asyncio.sleep(2)
            operation = client.operations.get(operation)

        imported_hashes[sha] = file_uploaded.name
        result.append(f"[{user}] Uploaded mới: {file.filename}")

    return result


# ---------------------------
# ASK ENDPOINT
# ---------------------------

@app.post("/ask")
async def ask_question(
    request: Request,
    question: str = Form(...),
    use_doc: str = Form("true")
):
    user = request.cookies.get("user")

    if not user:
        raise HTTPException(401, "Please log in!")

    if is_session_expired(user):
        active_sessions.pop(user, None)
        raise HTTPException(401, "Expired session!")

    update_last_active(user)

    active_sessions[user]["context"].append(question)

    # Build FULL CONVERSATION HISTORY 
    history = "\n".join(
        f"User: {q}" for q in active_sessions[user]["context"]
    )
    use_document = (use_doc.lower() == "true")

    if use_document:
        config = types.GenerateContentConfig(
            tools=[types.Tool(file_search=types.FileSearch(
                file_search_store_names=[store.name]
            ))]
        )
        instruction = f"""
            You should use the uploaded documents with File Search.
            """
    else:
        config = None
        instruction = f"""
            Do NOT use any uploaded documents.
            Answer only based on your own knowledge.
            """
    contents = f"""
        You are an AI assistant. With this history chat: {history} and my requirements: {instruction}, answer my quesions: {question}"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents= contents,
        config=config
    )

    return {"answer": response.text}

# ---------------------------
# FORENSICS ENDPOINT
# ---------------------------
# ---------------------------
# FORENSICS ENDPOINT (UPGRADED)
# ---------------------------
@app.post("/forensics")
async def forensic_analysis(file: UploadFile = File(...)):
    # 1. Validate Input
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        raise HTTPException(400, "Only JPG/PNG/WEBP allowed")

    # Tạo tên file unique để tránh trùng lặp khi nhiều user dùng
    request_id = str(uuid.uuid4())
    base_name = f"forensic_{request_id}"
    
    img_input_path = UPLOAD_DIR / f"{base_name}_orig.jpg"
    
    # Lưu file gốc
    content = await file.read()
    with open(img_input_path, "wb") as f:
        f.write(content)

    # ---------------------------
    # A. IMAGE PROCESSING (OpenCV/PIL)
    # ---------------------------
    
    # Load ảnh
    orig = Image.open(img_input_path).convert("RGB")
    img_np = cv2.imread(str(img_input_path)) 
    
    # 1. ELA (Error Level Analysis)
    # Nguyên lý: Lưu lại với chất lượng thấp, sau đó trừ đi ảnh gốc để xem sự khác biệt nén.
    temp_jpeg = UPLOAD_DIR / f"{base_name}_temp.jpg"
    img_ela_path = UPLOAD_DIR / f"{base_name}_ela.png"
    
    orig.save(temp_jpeg, "JPEG", quality=90)
    recompressed = Image.open(temp_jpeg).convert("RGB")
    ela = ImageChops.difference(orig, recompressed)
    
    # Tăng độ sáng ELA để dễ nhìn
    extrema = ela.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_enhanced = ImageEnhance.Brightness(ela).enhance(scale)
    ela_enhanced.save(img_ela_path)

    # 2. Noise Analysis (High Pass Filter)
    # Nguyên lý: Tách nhiễu hạt. Ảnh ghép thường có độ nhiễu không đồng nhất.
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    # Dùng filter Laplace để làm nổi bật cạnh và nhiễu
    noise_kernel = np.array([[0, -1,  0], [-1, 4, -1], [0, -1,  0]]) 
    noise = cv2.filter2D(gray, -1, noise_kernel)
    noise_norm = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX)
    img_noise_path = UPLOAD_DIR / f"{base_name}_noise.png"
    cv2.imwrite(str(img_noise_path), noise_norm)

    # 3. FFT (Fast Fourier Transform)
    # Nguyên lý: Phát hiện các mẫu lặp lại (thường thấy ở AI Deepfake hoặc ảnh bị resize/rotate)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    # Lấy log spectrum để hiển thị
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # Chuẩn hóa về 0-255 để lưu ảnh
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    img_fft_path = UPLOAD_DIR / f"{base_name}_fft.png"
    cv2.imwrite(str(img_fft_path), magnitude_spectrum)

    # Xóa file temp
    if temp_jpeg.exists():
        os.remove(temp_jpeg)

    # ---------------------------
    # B. AI ANALYSIS (GEMINI)
    # ---------------------------

    try:
        # Upload các ảnh lên Google GenAI (File API)
        # Lưu ý: Các file này chỉ tồn tại tạm thời trong project
        
        g_orig = client.files.upload(file=str(img_input_path), config=types.UploadFileConfig(display_name="Original Image"))
        g_ela = client.files.upload(file=str(img_ela_path), config=types.UploadFileConfig(display_name="ELA Map"))
        g_noise = client.files.upload(file=str(img_noise_path), config=types.UploadFileConfig(display_name="Noise Analysis"))
        g_fft = client.files.upload(file=str(img_fft_path), config=types.UploadFileConfig(display_name="FFT Spectrum"))

        # Đợi file active (thường ảnh nhỏ thì rất nhanh)
        while g_orig.state.name == "PROCESSING":
            time.sleep(1)
            g_orig = client.files.get(name=g_orig.name)

        # Prompt chuyên gia giám định
        prompt = """
        Bạn là một chuyên gia giám định hình ảnh kỹ thuật số (Digital Image Forensics Expert).
        Tôi cung cấp cho bạn 4 bức ảnh theo thứ tự:
        1. Ảnh gốc (Original Image).
        2. ELA (Error Level Analysis): Giúp phát hiện các vùng bị chỉnh sửa (splicing, healing) có mức nén khác biệt.
        3. Noise Analysis: Giúp phát hiện sự không đồng nhất về nhiễu hạt.
        4. FFT Spectrum: Giúp phát hiện các dấu hiệu của AI sinh ra (AI Generation artifacts) hoặc dấu vết chỉnh sửa hình học.

        NHIỆM VỤ CỦA BẠN:
        Hãy phân tích các hình ảnh này và đưa ra báo cáo ngắn gọn, dễ hiểu cho người dùng phổ thông.
        
        Cấu trúc trả lời:
        1. **Nhận định chung**: Ảnh này có khả năng cao là Thật (Authentic), Đã chỉnh sửa (Manipulated), hay do AI tạo ra (AI Generated)? Đưa ra tỉ lệ phần trăm tự tin.
        2. **Phân tích chi tiết**:
           - ELA: Có vùng nào sáng bất thường so với nền không? (Dấu hiệu ghép ảnh).
           - Noise: Nhiễu có đồng nhất không?
           - FFT: Có xuất hiện các ngôi sao (star patterns) hay lưới bất thường không? (Dấu hiệu AI/Deepfake).
        3. **Kết luận**: Tóm tắt lại phát hiện.

        Hãy trả lời bằng Tiếng Việt, giọng văn khách quan, chuyên nghiệp.
        """

        # Gọi Gemini
        response = client.models.generate_content(
            model="gemini-1.5-flash", # Dùng 1.5 Flash cho nhanh và tốt về Vision
            contents=[
                prompt,
                g_orig,
                g_ela,
                g_noise,
                g_fft
            ]
        )
        
        ai_analysis_text = response.text

    except Exception as e:
        ai_analysis_text = f"Không thể phân tích AI do lỗi: {str(e)}"

    # ---------------------------
    # C. RETURN RESPONSE
    # ---------------------------

    return {
        "analysis": ai_analysis_text,
        "images": {
            "original": f"/uploads/{base_name}_orig.jpg",
            "ela": f"/uploads/{base_name}_ela.png",
            "noise": f"/uploads/{base_name}_noise.png",
            "fft": f"/uploads/{base_name}_fft.png"
        }
    }