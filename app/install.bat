@echo off
setlocal
cd /d "%~dp0"
echo ============================================
echo   BirdCull installer  (needs internet, ~8 GB)
echo   This can take 10-20 minutes. Please wait.
echo ============================================
where uv >nul 2>nul
if errorlevel 1 (
  echo Installing helper tool...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)
where uv >nul 2>nul
if errorlevel 1 ( echo Could not install the helper tool. Check your internet and run this again. & pause & exit /b 1 )
uv venv --python 3.12 --allow-existing .venv || goto fail
echo Installing the AI engine for your graphics card...
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128 || goto fail
echo Installing the rest...
uv pip install --python .venv\Scripts\python.exe -r requirements.txt || goto fail
echo Downloading the bird-finder model...
.venv\Scripts\python.exe -c "from birdcull.core import ensure_model;print(ensure_model())" || goto fail
echo Checking the graphics card...
.venv\Scripts\python.exe -c "import torch;print('GPU found:' if torch.cuda.is_available() else 'NO GPU FOUND - it will still work but slowly.', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\BirdCull.lnk');$s.TargetPath='%~dp0run.bat';$s.WorkingDirectory='%~dp0';$s.Save()"
echo.
echo ===== Done! Use the BirdCull icon on your Desktop. =====
pause
exit /b 0
:fail
echo.
echo Something went wrong. Take a photo of this window and send it to your helper.
pause
exit /b 1
