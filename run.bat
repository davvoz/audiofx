@echo off
echo 🎵 Audio Visual FX Generator - Quick Start 🎵
echo =============================================

:: Controlla se Python è installato
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python non trovato! Installa Python da https://python.org
    pause
    exit /b 1
)

echo ✅ Python trovato!

:: Installa le dipendenze se necessarie
echo 📦 Controllando le dipendenze...
pip install -q numpy opencv-python pillow librosa soundfile scipy matplotlib moviepy ffmpeg-python

if errorlevel 1 (
    echo ❌ Errore nell'installazione delle dipendenze
    echo 💡 Prova: pip install -r requirements.txt
    pause
    exit /b 1
)

echo ✅ Dipendenze installate!

:: Menu opzioni
echo.
echo 🚀 Cosa vuoi fare?
echo 1. Esegui esempio rapido (genera audio e immagine di test)
echo 2. Usa i tuoi file (specifica audio e immagine)
echo 3. Solo installa dipendenze

set /p choice="Scegli opzione (1-3): "

if "%choice%"=="1" goto example
if "%choice%"=="2" goto custom
if "%choice%"=="3" goto end

:example
echo.
echo 🎬 Generando esempio con file di test...
python example.py
goto end

:custom
echo.
set /p audiofile="📢 Inserisci il path del file audio: "
set /p imagefile="🖼️ Inserisci il path dell'immagine: "
set /p outputfile="🎬 Nome video output [dark_techno_fx.mp4]: "

if "%outputfile%"=="" set outputfile=dark_techno_fx.mp4

echo.
echo 🚀 Generando il tuo video dark techno...
python audio_visual_fx.py --audio "%audiofile%" --image "%imagefile%" --output "%outputfile%"

:end
echo.
echo 🔥 Processo completato! 🔥
echo.
echo 💡 Per uso avanzato:
echo    python audio_visual_fx.py --help
echo.
pause