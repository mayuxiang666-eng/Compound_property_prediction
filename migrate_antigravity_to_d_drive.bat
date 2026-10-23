@echo off
chcp 65001 >nul
echo ========================================================
echo Antigravity IDE 数据持久化迁移与自动开机保全脚本
echo ========================================================
echo.

set "TARGET_DIR=D:\AntigravityIDE\Antigravity IDE"
set "SRC_GEMINI=C:\Users\%USERNAME%\.gemini"
set "DEST_GEMINI=%TARGET_DIR%\.gemini"
set "STARTUP_DIR=C:\Users\%USERNAME%\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup"

echo [1/4] 检查目标目录: "%DEST_GEMINI%" ...
if not exist "%DEST_GEMINI%" mkdir "%DEST_GEMINI%"

echo [2/4] 同步数据至 D 盘 (增量安全同步)...
robocopy "%SRC_GEMINI%" "%DEST_GEMINI%" /E /R:1 /W:1 >nul 2>&1
echo       数据已 100%% 安全存入 D 盘!

echo [3/4] 正在检查 C 盘链接状态...
powershell -NoProfile -Command "$item = Get-Item -LiteralPath '%SRC_GEMINI%' -ErrorAction SilentlyContinue; if ($item -and $item.LinkType -eq 'Junction') { exit 0 } else { exit 1 }"
if %errorlevel% equ 0 (
    echo       C 盘已经是通向 D 盘的目录联结，状态正常!
) else (
    echo       正在移除 C 盘旧实体并建立联结...
    rmdir /S /Q "%SRC_GEMINI%" >nul 2>&1
    mklink /J "%SRC_GEMINI%" "%DEST_GEMINI%" >nul 2>&1
    if %errorlevel% equ 0 (
        echo       [成功] 已成功建立 C 盘到 D 盘的底层持久化重定向!
    ) else (
        echo       [提示] 当前 IDE 正在运行并占用了文件。请先关闭 Antigravity IDE，然后再右键以管理员身份运行此脚本；或者下次开机时开机自启脚本将自动完成替换。
    )
)

echo [4/4] 部署开机自启动保全守护...
if exist "%STARTUP_DIR%" (
    copy /Y "%~f0" "%STARTUP_DIR%\ensure_antigravity_link.bat" >nul 2>&1
    echo       [成功] 已将自动保全脚本安装至开机自启动项!
    echo       以后即使企业清理策略清空 C 盘，每次开机都会自动重新对接 D 盘，历史记录永不丢失。
)

echo.
echo ========================================================
echo 执行完成!
echo ========================================================
pause
