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
@app.post("/forensics")
async def forensic_analysis(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(400, "Only JPG/PNG allowed")

    img_input = UPLOAD_DIR / "forensic_input.jpg"
    with open(img_input, "wb") as f:
        f.write(await file.read())

    orig = Image.open(img_input).convert("RGB")

    temp_jpeg = UPLOAD_DIR / "forensic_recompressed.jpg"
    img_ela = UPLOAD_DIR / "forensic_ela.png"

    orig.save(temp_jpeg, "JPEG", quality=90)
    recompressed = Image.open(temp_jpeg).convert("RGB")
    ela = ImageChops.difference(orig, recompressed)
    ela_enhanced = ImageEnhance.Brightness(ela).enhance(15)
    ela_enhanced.save(img_ela)

    img_np = cv2.imread(str(img_input)) 
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    kernel = np.array([[0, -1,  0],
                       [-1, 4, -1],
                       [0, -1,  0]])

    noise = cv2.filter2D(gray, -1, kernel)
    noise_norm = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX)
    img_noise = UPLOAD_DIR / "forensic_noise.png"
    cv2.imwrite(str(img_noise), noise_norm)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    spectrum = 20 * np.log(np.abs(fshift) + 1)

    img_fft = UPLOAD_DIR / "forensic_fft.png"
    cv2.imwrite(str(img_fft), spectrum)

    return {
        "message": "Forensic analysis complete",
        "original": "/uploads/forensic_input.jpg",
        "ela": "/uploads/forensic_ela.png",
        "noise": "/uploads/forensic_noise.png",
        "fft": "/uploads/forensic_fft.png"
    }
