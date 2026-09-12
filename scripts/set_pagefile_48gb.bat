@echo off
:: Batch script to set Windows Pagefile to 32GB - 48GB (Run as Administrator)
echo ===================================================
echo Configuring Windows Pagefile for LTX-2.5 Workstation
echo ===================================================

net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] This script must be run as Administrator!
    echo Please right-click this file and select "Run as administrator".
    echo.
    pause
    exit /b 1
)

echo [*] Disabling automatic pagefile management...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$sys = Get-CimInstance Win32_ComputerSystem; $sys.AutomaticManagedPagefile = $false; Set-CimInstance -InputObject $sys"

echo [*] Configuring C:\pagefile.sys: Initial 32768 MB (32 GB), Maximum 49152 MB (48 GB)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management' -Name 'PagingFiles' -Value @('C:\pagefile.sys 32768 49152')"

echo.
echo ===================================================
echo [+] SUCCESS: Pagefile configured to 32 GB - 48 GB!
echo [+] Total commit limit will expand to ~96 GB - 112 GB.
echo Note: A system reboot is recommended for Windows to
echo       commit the new pagefile allocations.
echo ===================================================
pause
