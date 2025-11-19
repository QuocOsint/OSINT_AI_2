from google import genai
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import time, os
import uuid 
from pathlib import Path
import hashlib
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
'''
Overview features: 
- Upload any content types
- Store it in a Gemini File Search Store 
- Ask questions about file uploaded
- Return model response real-time

'''
imported_hashes = {}
def file_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()

# Initilize Gemini client
client = genai.Client(api_key = "AIzaSyC1vj7Cg-YeABOyIO9WO16DHUlBG1k0c2E")
store_name = "Proof-of-Content"
# Check if that store is already created 
store = None
stores = client.file_search_stores.list()
for s in stores:
    if s.display_name == store_name: 
        print("Reuse store: ", s.name)
        store = s
        break
if not store:
    store = client.file_search_stores.create(config = {"display_name" : "Proof-of-Content"})

# Create app
app = FastAPI(title = "Gemini File Search POC")

@app.get("/",response_class=HTMLResponse)
async def home():
    with open("static/index.html", "r", encoding = "utf-8") as f:
        return f.read()

@app.post("/upload")
# Upload file to API Files --> Gemini File Search Store
async def upload_file(files: list[UploadFile] = File(...)):
    imported_files = []
    # Open file 
    for file in files: 
        ext = Path(file.filename).suffix
        safe_name = f"{uuid.uuid4()}{ext}"
        file_path = UPLOAD_DIR / safe_name
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Upload to Files API and check duplicated content with hash
        sha = file_hash(file_path)
        if sha in imported_hashes:
            # imported_files.append({
            #     "original_filename": file.filename,
            #     "status": "duplicate",
            #     "duplicate_of": imported_hashes[sha]
            # })
            continue

        file_uploaded = client.files.upload(file=str(file_path))
        # Import file into store
        operation = client.file_search_stores.import_file(
            file_search_store_name=store.name,
            file_name= file_uploaded.name
        )
        print("Importing files...")
        while not operation.done:
            time.sleep(5)
            operation = client.operations.get(operation)
        
        imported_hashes[sha] = file_uploaded.name
        # imported_files.append({
        #     "filename": file.filename,
        #     "saved_as": safe_name,
        #     "hash": sha,
        #     "path": str(file_path),
        #     "file_id": file_uploaded.name
        # })
    return {"message": f"All files uploaded!"}

@app.post("/ask")
async def ask_question(question: str =  Form(...)):
    response = client.models.generate_content(
        model = "gemini-2.5-flash",
        contents = question, 
        config = {
            "tools": [
                {"file_search":{"file_search_store_names": [store.name]}}
            ]
        }
    )
    return JSONResponse({"answer": response.text})
