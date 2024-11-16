@echo off
title Minnsun's Bow Client Setup

:: Kiểm tra xem Python đã được cài đặt chưa
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python chua duoc cai dat
    echo tai python tai https://www.python.org/downloads/
    pause
    exit /b
)

:: Kiểm tra xem pip đã được cài đặt chưa
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Khong tim thay pip
    echo Dang cai dat pip...
    python -m ensurepip --upgrade
    if %errorlevel% neq 0 (
        echo Co loi, vui long tai thu cong https://pip.pypa.io/en/stable/installation/
        pause
        exit /b
    )
    echo cai pip thanh cong
)


echo Dang tao moi truong python ao
python -m venv venv

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo Khong the tao moi truong ao
    pause
    exit /b
)

:: Nâng cấp pip
echo Tien hanh nang cap pip
pip install --upgrade pip

:: Cài đặt các thư viện cần thiết từ requirements.txt
if exist "requirements.txt" (
    echo Dang cai dat cac thu vien yeu cau
    pip install psutil numpy requests py-cpuinfo psycopg2-binary pyopencl siphash24 GPUtil cupy
    if %errorlevel% neq 0 (
        echo May tinh cua ban co ve khong phu hop voi thu vien nao do, vui long kiem tra trong requirements.txt
        pause
        exit /b
    )
) else (
    echo khong tim thay file requirements.txt
    pause
    exit /b
)

:: Chạy chương trình Python
echo Oce bắt đầu chạy Minnsun's Bow Client...
python client.py

:: Thông báo hoàn tất
echo Cam on ban
pause
exit
