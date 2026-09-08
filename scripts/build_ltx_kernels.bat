@echo off
setlocal enabledelayedexpansion

echo ======================================================================
echo   COMPILING LTX-KERNELS FOR NVIDIA RTX 5080 (BLACKWELL SM 12.0)
echo ======================================================================

set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo [!] ERROR: vcvars64.bat not found at "%VCVARS%"
    exit /b 1
)

echo [+] Initializing MSVC x64 Developer Environment...
call "%VCVARS%"
if errorlevel 1 (
    echo [!] ERROR: Failed to initialize MSVC environment.
    exit /b 1
)

set "CUDA_PATH=C:\Users\jerry\cuda_12.8"
set "CUDA_HOME=C:\Users\jerry\cuda_12.8"
set "PATH=%CUDA_PATH%\bin;%PATH%"
set "TORCH_CUDA_ARCH_LIST=12.0"
set "DISTUTILS_USE_SDK=1"
set "MAX_JOBS=1"

echo [+] Verifying nvcc compiler:
nvcc --version
if errorlevel 1 (
    echo [!] ERROR: nvcc failed to run.
    exit /b 1
)

echo [+] Verifying cl compiler:
cl
if errorlevel 1 (
    rem cl prints logo and returns errorlevel 2 if no args, which is expected
)

echo [+] Compiling ltx-kernels in editable mode...
C:\Users\jerry\.local\bin\uv.exe pip install -e packages/ltx-kernels --no-build-isolation --python .\.venv\Scripts\python.exe -v

if errorlevel 1 (
    echo [!] ERROR: ltx-kernels build failed.
    exit /b 1
)

echo [+] Successfully compiled ltx-kernels!
echo ======================================================================
exit /b 0
