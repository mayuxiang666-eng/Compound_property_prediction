# Windows Server 生产环境详细上线部署指南

本指南专为 **Windows Server 环境** 编写，详细说明需要拷贝到服务器的目录与文件清单、服务器前置依赖软件安装、NSSM 系统服务注册、计划任务配置及上线验证步骤。

---

## 1. 需要拷贝到服务器的文件与目录清单 (File Transfer Checklist)

请将本地项目中的以下**核心目录与文件**打包，拷贝至服务器的目标路径（建议服务器目录为 `C:\Compound_property_prediction\`）：

### 必须拷贝的文件与文件夹 (MUST Copy)

| 目录/文件名 | 类型 | 作用与说明 |
| :--- | :---: | :--- |
| **`server_deployment/`** | 文件夹 | **包含全套服务器部署、NSSM注册、风险预警与每日定时运行代码**：<br>• `daily_pipeline_runner.py`<br>• `daily_export_parquet.py`<br>• `model_risk_alerter.py`<br>• `requirements.txt`<br>• `start_mlflow_server.bat`<br>• `nssm_install_services.bat`<br>• `schedule_daily_task.bat` |
| **`Mooney_Prediction_Pipeline/`** | 文件夹 | **包含预训练模型 Bundle、SQL 配置文件及特征抽取核心代码**：<br>• `models/` (已训练好的模型 joblib bundle 与组数据集)<br>• `data_processing/credentials.json` (数据库连接凭据)<br>• `data_processing/function_definitions_M1.py`<br>• `data_processing/pipeline_orchestrator.py` |
| **`data_store/`** | 文件夹 | **固定 Parquet 数据仓及动态状态保存目录**：<br>• `parquet/` (Parquet 列式存储路径)<br>• `rolling_calibration_state.json` (滚动校准状态持久化文件)<br>• `model_risk_alerts_log.csv` (风险预警日志表) |
| **`stage_statistics_enriched_all_features_weather_v4.csv`** | 文件 | 历史胶料基线特征库 (用于配方 Lookup 与基线比对) |
| **`stage_statistics_enriched.csv`** | 文件 | 历史高频温度曲线库 (用于物理动力学积分计算) |
| **`oilload_mapping_master_batch.csv`** | 文件 | 母胶含油量与炭黑分类主映射表 |
| **`mlflow.db`** | 文件 | MLflow SQLite 后端数据库文件 |

---

### 无需拷贝的文件与文件夹 (EXCLUDE / Do NOT Copy)

> [!CAUTION]
> **请勿拷贝以下虚拟环境及临时垃圾文件**，防止服务器环境冲突或传输文件过大：
> - `.venv/` / `venv/` (本地 Python 虚拟环境，服务器端重新安装依赖)
> - `.git/` / `.gitignore`
> - `.vscode/` / `.idea/`
> - `__pycache__/` (Python 字节码编译缓存)
> - `catboost_info/` / `scratch/` (本地临时运行调试日志)

---

## 2. 服务器部署前置软件与依赖准备 (Prerequisites)

在服务器上运行之前，请确认已安装以下 4 个前置组件：

1. **Python 3.10+ (推荐 3.12)**:
   - 下载并安装 Python Windows Installer。
   - **勾选 `Add Python to PATH`**（确保 `python` 命令可在 cmd / PowerShell 全局调用）。
2. **NSSM.exe (Non-Sucking Service Manager)**:
   - 从 [NSSM 官网下载](https://nssm.cc/download) 解压 `nssm.exe`（64 位版本）。
   - 将 `nssm.exe` 放置在服务器的 `C:\Windows\System32\` 目录下（或直接存放在项目 `server_deployment/` 文件夹中）。
3. **ODBC Driver 17 for SQL Server**:
   - 服务器需安装 [Microsoft ODBC Driver 17 for SQL Server](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server)，以支持 `pyodbc` 连接 MMS `SFEPLANT` 数据库。
4. **服务器防火墙 9999 端口放行**:
   - 在 Windows Server 防火墙中添加“入站规则”，放行 **TCP 9999 端口**，以便终端用户能在浏览器访问 `http://<服务器IP>:9999` 查看 MLflow 看板。

---

## 3. 服务器端一步步上线部署步骤 (Step-by-Step Installation)

请在服务器上使用**管理员权限的 cmd 或 PowerShell** 依次执行以下步骤：

### 步骤 1: 将打包好的项目解压至服务器目录
假设解压后的完整路径为：
`C:\Compound_property_prediction\`

打开 cmd 切换到该目录：
```cmd
cd /d C:\Compound_property_prediction
```

---

### 步骤 2: 安装服务器 Python 依赖包
执行以下命令，一键安装项目所需全部 Python 依赖（含 MLflow 3.14.0, PyArrow, LightGBM 等）：
```cmd
python -m pip install --upgrade pip
python -m pip install -r server_deployment/requirements.txt
```

---

### 步骤 3: 使用 NSSM 注册并启动 MLflow Web 9999 端口系统服务
鼠标右键点击 `server_deployment/nssm_install_services.bat`，选择**“以管理员身份运行”**。

或者在管理员 cmd 中手动执行：
```cmd
cd /d C:\Compound_property_prediction
nssm install MLflowWebServer "python.exe" "-m mlflow server --host 0.0.0.0 --port 9999 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns"
nssm set MLflowWebServer AppDirectory "C:\Compound_property_prediction"
nssm set MLflowWebServer DisplayName "Mooney Viscosity MLflow UI (Port 9999)"
nssm set MLflowWebServer Start SERVICE_AUTO_START
nssm start MLflowWebServer
```

---

### 步骤 4: 注册 Windows 计划任务 (每日 03:00 AM 自动运行)
鼠标右键点击 `server_deployment/schedule_daily_task.bat`，选择**“以管理员身份运行”**。

或者在管理员 cmd 中手动执行：
```cmd
schtasks /create /tn "MooneyDailyPipeline" /tr "python \"C:\Compound_property_prediction\server_deployment\daily_pipeline_runner.py\"" /sc daily /st 03:00 /f /ru "SYSTEM"
```

---

## 4. 上线验证与健康检查 (Health Check)

部署完成后，请在服务器上进行以下 3 项验证：

1. **检查 MLflow 前端**:
   打开浏览器访问：`http://localhost:9999` 或 `http://<服务器IP>:9999`。
   - 确认能看到 `With-Oil Carbon-Black Track` 实验面板。
2. **测试手动干跑管道**:
   在 cmd 中执行：
   ```cmd
   python server_deployment/daily_pipeline_runner.py
   ```
   - 确认控制台输出 `DAILY PIPELINE COMPLETED & LOGGED TO MLFLOW (http://localhost:9999)`。
   - 确认 `data_store/parquet/2026-07/` 下生成了 Parquet 文件。
3. **检查风险预警日志**:
   查看 `data_store/model_risk_alerts_log.csv` 文件，确认风险预警引擎正常记录。

---

## 5. 常见故障排查 (Troubleshooting)

* **问题 1: `nssm 不是内部或外部命令`**
  * 解决: 请确保将 `nssm.exe` 复制到了 `C:\Windows\System32\` 目录。
* **问题 2: pyodbc 连接数据库失败**
  * 解决: 确认服务器已安装 `ODBC Driver 17 for SQL Server`，且服务器能 Ping 通 MMS 数据库 IP 地址。
* **问题 3: 浏览器打不开 9999 端口页面**
  * 解决: 运行 `nssm status MLflowWebServer` 确认服务运行正常，并在 Windows 防火墙中放行 9999 入站端口。
