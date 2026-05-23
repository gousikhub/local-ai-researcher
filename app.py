import os
from typing import List
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from engine import AIResearcherEngine

app = FastAPI()
ai_engine = AIResearcherEngine()

os.makedirs("temp_uploads", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Researcher | Enterprise</title>
        <style>
            :root { --bg: #0b0f19; --panel: #111827; --primary: #3b82f6; --primary-hover: #2563eb; --border: #1f2937; --text: #f3f4f6; --text-muted: #9ca3af; }
            * { box-sizing: border-box; font-family: 'Inter', -apple-system, sans-serif; }
            body { margin: 0; display: flex; height: 100vh; background-color: var(--bg); color: var(--text); overflow: hidden; }
            
            ::-webkit-scrollbar { width: 6px; }
            ::-webkit-scrollbar-thumb { background: #374151; border-radius: 10px; }
            
            .sidebar { width: 380px; background: var(--panel); border-right: 1px solid var(--border); padding: 24px; display: flex; flex-direction: column; gap: 24px; overflow-y: auto; }
            .header h1 { font-size: 1.2rem; font-weight: 600; margin: 0; letter-spacing: 0.5px; }
            .header span { font-size: 0.75rem; color: #10b981; font-weight: bold; background: rgba(16, 185, 129, 0.1); padding: 2px 8px; border-radius: 12px; }
            
            .drop-zone { border: 2px dashed #374151; border-radius: 12px; padding: 30px 20px; text-align: center; cursor: pointer; transition: all 0.2s ease; background: rgba(17, 24, 39, 0.5); }
            .drop-zone:hover, .drop-zone.dragover { border-color: var(--primary); background: rgba(59, 130, 246, 0.05); }
            .drop-zone svg { width: 40px; height: 40px; fill: var(--text-muted); margin-bottom: 10px; transition: fill 0.2s; }
            .drop-zone:hover svg { fill: var(--primary); }
            .drop-zone p { margin: 0; font-size: 0.9rem; font-weight: 500; }
            .drop-zone span { font-size: 0.8rem; color: var(--text-muted); }
            #fileInput { display: none; }
            .selected-file { margin-top: 10px; font-size: 0.85rem; color: #10b981; font-weight: 500; display: none; white-space: pre-wrap; word-break: break-all;}

            .input-group { display: flex; flex-direction: column; gap: 8px; }
            .input-group label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); font-weight: 600; }
            input[type="text"], textarea { background: var(--bg); border: 1px solid var(--border); color: white; padding: 12px; border-radius: 8px; font-size: 0.9rem; outline: none; transition: border 0.2s; }
            input[type="text"]:focus, textarea:focus { border-color: var(--primary); }
            
            button { background: var(--primary); color: white; border: none; padding: 12px; border-radius: 8px; font-weight: 600; cursor: pointer; transition: all 0.2s; display: flex; justify-content: center; align-items: center; gap: 8px; }
            button:hover:not(:disabled) { background: var(--primary-hover); transform: translateY(-1px); }
            button:disabled { opacity: 0.7; cursor: not-allowed; }
            .btn-danger { background: transparent; border: 1px solid #ef4444; color: #ef4444; }
            .btn-danger:hover { background: #ef4444; color: white; }

            .main-chat { flex-grow: 1; display: flex; flex-direction: column; background: var(--bg); }
            .chat-history { flex-grow: 1; padding: 40px; overflow-y: auto; display: flex; flex-direction: column; gap: 24px; }
            .message { max-width: 80%; padding: 16px 20px; border-radius: 12px; line-height: 1.6; font-size: 0.95rem; }
            .msg-user { background: var(--primary); align-self: flex-end; border-bottom-right-radius: 4px; box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2); }
            .msg-ai { background: var(--panel); border: 1px solid var(--border); align-self: flex-start; border-bottom-left-radius: 4px; color: #e5e7eb; }
            
            .chat-input-wrapper { padding: 20px 40px; background: var(--bg); border-top: 1px solid var(--border); }
            .chat-input-box { display: flex; gap: 12px; background: var(--panel); padding: 8px; border-radius: 12px; border: 1px solid var(--border); }
            .chat-input-box input { flex-grow: 1; border: none; background: transparent; font-size: 1rem; padding: 0 10px; color: white; outline: none;}
            .chat-input-box button { margin: 0; padding: 12px 24px; }

            #toast { position: fixed; bottom: -100px; right: 30px; background: #10b981; color: white; padding: 12px 24px; border-radius: 8px; font-weight: 500; box-shadow: 0 10px 25px rgba(0,0,0,0.2); transition: bottom 0.4s cubic-bezier(0.18, 0.89, 0.32, 1.28); z-index: 1000; }
            
            .spinner { border: 3px solid rgba(255,255,255,0.3); border-radius: 50%; border-top: 3px solid white; width: 16px; height: 16px; animation: spin 1s linear infinite; display: none; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        </style>
    </head>
    <body>

        <div class="sidebar">
            <div class="header">
                <h1>Neural Engine Core</h1>
                <span>MULTI-DOC MODE</span>
            </div>

            <div class="input-group">
                <label>1. Inject Documents (Select Multiple)</label>
                <div class="drop-zone" id="dropZone">
                    <svg viewBox="0 0 24 24"><path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.36 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM14 13v4h-4v-4H7l5-5 5 5h-3z"/></svg>
                    <p>Drag & drop PDFs/TXTs here</p>
                    <span>You can select multiple files</span>
                    <input type="file" id="fileInput" accept=".pdf,.txt,.csv,.json" multiple>
                </div>
                <div class="selected-file" id="selectedFileName"></div>
                <button id="uploadBtn" onclick="uploadFile()">
                    <div class="spinner" id="uploadSpinner"></div>
                    <span id="uploadBtnText">Process All Documents</span>
                </button>
            </div>

            <div class="input-group" style="margin-top: 10px;">
                <label>2. Inject Raw Text</label>
                <input type="text" id="manualLabel" placeholder="e.g., God of War Lore">
                <textarea id="manualText" rows="3" placeholder="Paste data here..."></textarea>
                <button id="textBtn" onclick="submitManualData()">Inject Knowledge</button>
            </div>
            
            <div style="flex-grow: 1;"></div>
            <button class="btn-danger" onclick="clearSystem()">Purge System Memory</button>
        </div>

        <div class="main-chat">
            <div class="chat-history" id="chatHistory">
                <div class="message msg-ai">Neural Engine online. Multi-document cross-referencing enabled. Waiting for data injections...</div>
            </div>

            <div class="chat-input-wrapper">
                <div class="chat-input-box">
                    <input type="text" id="questionInput" placeholder="Ask a question across all injected data..." onkeypress="if(event.key === 'Enter') askQuestion()">
                    <button id="askBtn" onclick="askQuestion()">
                        <div class="spinner" id="askSpinner"></div>
                        <span id="askBtnText">Execute</span>
                    </button>
                </div>
            </div>
        </div>

        <div id="toast">Message here</div>

        <script>
            function showToast(message, isError = false) {
                const toast = document.getElementById('toast');
                toast.innerText = message;
                toast.style.background = isError ? '#ef4444' : '#10b981';
                toast.style.bottom = '30px';
                setTimeout(() => { toast.style.bottom = '-100px'; }, 3000);
            }

            function appendMessage(sender, text) {
                const history = document.getElementById('chatHistory');
                const msgDiv = document.createElement('div');
                msgDiv.className = sender === 'user' ? 'message msg-user' : 'message msg-ai';
                msgDiv.innerText = text;
                history.appendChild(msgDiv);
                history.scrollTop = history.scrollHeight;
            }

            function setLoading(btnId, isLoading) {
                const btn = document.getElementById(btnId);
                const spinner = document.getElementById(btnId.replace('Btn', 'Spinner'));
                const text = document.getElementById(btnId.replace('Btn', 'BtnText'));
                btn.disabled = isLoading;
                spinner.style.display = isLoading ? 'block' : 'none';
                text.style.display = isLoading ? 'none' : 'block';
            }

            const dropZone = document.getElementById('dropZone');
            const fileInput = document.getElementById('fileInput');
            const fileNameDisplay = document.getElementById('selectedFileName');

            dropZone.addEventListener('click', () => fileInput.click());
            
            dropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                dropZone.classList.add('dragover');
            });
            
            dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
            
            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.classList.remove('dragover');
                if (e.dataTransfer.files.length) {
                    fileInput.files = e.dataTransfer.files;
                    updateFileDisplay();
                }
            });

            fileInput.addEventListener('change', updateFileDisplay);

            function updateFileDisplay() {
                if(fileInput.files.length) {
                    if(fileInput.files.length === 1) {
                        fileNameDisplay.innerText = "Selected: " + fileInput.files[0].name;
                    } else {
                        fileNameDisplay.innerText = "Selected " + fileInput.files.length + " files ready for processing.";
                    }
                    fileNameDisplay.style.display = 'block';
                }
            }

            async function uploadFile() {
                if(fileInput.files.length === 0) return showToast("Please select files first!", true);
                
                let formData = new FormData();
                // LOOP THROUGH ALL FILES AND APPEND THEM
                for(let i = 0; i < fileInput.files.length; i++) {
                    formData.append("files", fileInput.files[i]);
                }
                
                setLoading('uploadBtn', true);
                try {
                    // CHANGED: Using relative path '/upload'
                    let res = await fetch('/upload', { method: 'POST', body: formData });
                    let data = await res.json();
                    showToast(data.message);
                    appendMessage('ai', "[System] " + data.message);
                    fileInput.value = "";
                    fileNameDisplay.style.display = 'none';
                } catch(e) {
                    showToast("Error uploading files.", true);
                }
                setLoading('uploadBtn', false);
            }

            async function submitManualData() {
                let lbl = document.getElementById('manualLabel').value;
                let txt = document.getElementById('manualText').value;
                if(!lbl || !txt) return showToast("Fill both label and content.", true);

                let formData = new FormData();
                formData.append("label", lbl);
                formData.append("text", txt);

                let btn = document.getElementById('textBtn');
                btn.innerText = "Injecting..."; btn.disabled = true;
                
                try {
                    // CHANGED: Using relative path '/manual'
                    let res = await fetch('/manual', { method: 'POST', body: formData });
                    let data = await res.json();
                    
                    btn.innerText = "Inject Knowledge"; btn.disabled = false;
                    showToast(data.message);
                    appendMessage('ai', "[System] " + data.message);
                    document.getElementById('manualLabel').value = "";
                    document.getElementById('manualText').value = "";
                } catch(e) {
                    btn.innerText = "Inject Knowledge"; btn.disabled = false;
                    showToast("Error injecting text.", true);
                }
            }

            async function askQuestion() {
                let qInput = document.getElementById('questionInput');
                let q = qInput.value.trim();
                if(!q) return;

                appendMessage('user', q);
                qInput.value = "";
                
                setLoading('askBtn', true);
                let formData = new FormData();
                formData.append("question", q);

                try {
                    // CHANGED: Using relative path '/ask'
                    let res = await fetch('/ask', { method: 'POST', body: formData });
                    let data = await res.json();
                    appendMessage('ai', data.answer);
                } catch(e) {
                    appendMessage('ai', "[System Error] Backend connection failed.");
                }
                setLoading('askBtn', false);
            }

            async function clearSystem() {
                if(!confirm("Purge all data from RAM?")) return;
                try {
                    // CHANGED: Using relative path '/clear'
                    await fetch('/clear', {method: 'POST'});
                    document.getElementById('chatHistory').innerHTML = '<div class="message msg-ai">Memory purged. Awaiting new data...</div>';
                    showToast("System Memory Cleared");
                } catch(e) {
                    showToast("Error clearing memory.", true);
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/clear")
def clear_memory():
    ai_engine.clear_database()
    return {"message": "Memory wiped."}

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    success_count = 0
    for file in files:
        temp_path = f"temp_uploads/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        success = ai_engine.parse_file(temp_path)
        if success:
            success_count += 1
            
        if os.path.exists(temp_path): 
            os.remove(temp_path)
            
    if success_count > 0:
        return {"message": f"Successfully injected {success_count} file(s) into the neural matrix!"}
    return {"message": "Error processing files."}

@app.post("/manual")
def upload_manual(label: str = Form(...), text: str = Form(...)):
    ai_engine.add_manual_text(label, text)
    return {"message": f"Raw text '{label}' injected into memory."}

@app.post("/ask")
def query_ai(question: str = Form(...)):
    return {"answer": ai_engine.ask(question)}