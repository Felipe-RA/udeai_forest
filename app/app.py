from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from pydantic import BaseModel
from fastapi import Form
import subprocess
import os
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class VegetationRequest(BaseModel):
    data_dir: str
    path_to_model: str

async def common_parameters(data_dir: str = Form(None), path_to_model: str = Form(None), body: VegetationRequest = None):
    if body:
        data_dir = body.data_dir
        path_to_model = body.path_to_model
    return data_dir, path_to_model

@app.post("/generate-map/")
async def generate_map(params: tuple = Depends(common_parameters)):
    data_dir, path_to_model = params
    if not data_dir or not path_to_model:
        raise HTTPException(status_code=400, detail="Missing data directory or model path")
    
    data_dir, path_to_model = (f"./{data_dir}", f"./{path_to_model}")

    # Check if paths are valid
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        raise HTTPException(status_code=404, detail=f"Data directory not found: {data_dir}")
    if not os.path.exists(path_to_model):
        logger.error(f"Model path not found: {path_to_model}")
        raise HTTPException(status_code=404, detail=f"Model path not found: {path_to_model}")

    # Construct the command to run the Python script
    command = f"/app/venv/bin/python ./generate_vegetation_map.py {data_dir} {path_to_model}"
    logger.debug(f"Running command: {command}")
    try:
        subprocess.run(command, check=True, shell=True)
        map_directory = f'html_maps/{os.path.basename(data_dir)}'
        return HTMLResponse(content=f"""
            <html>
                <head>
                    <title>Map Generation</title>
                </head>
                <body>
                    <h1>Map generation initiated successfully.</h1>
                    <p>Check <a href="/view-map/{os.path.basename(data_dir)}">{map_directory}/vegetation_map.html</a> for the output.</p>
                    <p style="color: red;">Redirection in 5 seconds or <a href="/view-map/{os.path.basename(data_dir)}">click here</a> to be redirected.</p>
                    <script>
                        setTimeout(function() {{
                            window.location.href = "/view-map/{os.path.basename(data_dir)}";
                        }}, 5000);
                    </script>
                </body>
            </html>
        """, status_code=200)
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/view-map/{data_dir}", response_class=HTMLResponse)
async def view_map(data_dir: str):
    map_path = f"html_maps/{data_dir}/vegetation_map.html"
    if os.path.exists(map_path):
        return FileResponse(map_path, media_type='text/html')
    else:
        raise HTTPException(status_code=404, detail="Map not found at "+ map_path)


@app.get("/form", response_class=HTMLResponse)
async def form():
    html_content = """
    <html>
        <head>
            <title>Generate Map</title>
            <script>
                function toggleLock() {
                    var modelPathInput = document.getElementById("path_to_model");
                    var lockCheckbox = document.getElementById("lock_model_path");
                    modelPathInput.readOnly = !lockCheckbox.checked;
                }
            </script>
        </head>
        <body>
            <h1>Generate Map</h1>
            <form id="generate-map-form" action="/generate-map/" method="post">
                <label for="data_dir">Data Directory:</label>
                <input type="text" id="data_dir" name="data_dir" required><br><br>
                <label for="path_to_model">Model Path:</label>
                <input type="text" id="path_to_model" name="path_to_model" value="src/trained_models/VGGUdeaSpectral1/VGGUdeaSpectral_model1.joblib" readonly required>
                <input type="checkbox" id="lock_model_path" checked onclick="toggleLock()"> Lock
                <br><br>
                <input type="submit" value="Submit">
            </form>
            <div id="loading-message" style="display:none;">
                <p>Please wait, this process may take about 15 minutes...</p>
                <img src="https://i.gifer.com/ZZ5H.gif" alt="Loading spinner" />
            </div>
            <script>
                document.getElementById("generate-map-form").onsubmit = function() {
                    document.getElementById("generate-map-form").style.display = "none";
                    document.getElementById("loading-message").style.display = "block";
                };
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)
