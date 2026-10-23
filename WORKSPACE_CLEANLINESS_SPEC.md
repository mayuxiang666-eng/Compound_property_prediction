# 工作区临时文件管理与清洁度规范
# Workspace Cleanliness & Temporary File Management Specification

> **强制效力**：所有在本项目中工作的 AI Agent、自动化脚本以及开发人员必须严格遵循本规范。违反应受治理并立即纠正。

---

## 一、核心红线原则

1. **零污染原则（Zero Pollution in Workspace Root）**
   * **严禁**直接在工作区根目录下创建无计划、随意的探查脚本（如 `test_*.py`、`inspect_*.py`、`explore_*.py`、`check_*.py`、`tmp_*` 等）。
   * 工作区根目录只允许存放经过评审、具备长期维护价值的**核心工程模块**（如 `db_client.py`、`compound_tracer.py`）、**配置文件**（如 `db_config.json`）及**必要工程文档**。

2. **即创即销原则（Immediate Cleanup Upon Completion）**
   * 排查、探索、验证是手段，不是终点。一旦完成特定逻辑的排查或结论验证，**必须在当前会话结束前主动将所有临时生成的文件彻底删除**。
   * 严禁将临时文件遗留给后续开发或用户视线。

---

## 二、临时验证的标准操作路径

若在排查复杂问题时确需编写代码验证，必须按照以下优先级进行：

### 优先级 1：优先采用内存执行（免文件落地）
能用命令行单行代码解决的验证，坚决不落地任何文件。
```powershell
# 推荐做法：通过 python -c 执行内存脚本
python -c "import db_client; print(db_client.query_df('SELECT @@VERSION'))"
```

### 优先级 2：若必须落地脚本，统一隔离在专属临时目录
如果逻辑复杂、多行代码无法在一行命令中完成：
1. 必须将脚本创建在专属沙盒目录（如 `.temp/` 或系统沙盒 `scratch/` 目录中），不得散落在代码仓库各处。
2. 命名清晰标明临时性质（如 `.temp/tmp_scratch_test.py`）。

### 优先级 3：验证结束必须执行物理删除
在向上级或用户汇报结论前，执行清理命令清除所有临时文件：
```powershell
Remove-Item -Path .temp -Recurse -Force
```

---

## 三、文件留存审核清单（保留 vs 清理）

| 文件类别 | 示例 | 处置策略 |
| :--- | :--- | :--- |
| **核心业务模块** | `db_client.py`, `compound_tracer.py` | ✅ **长期保留**（有明确业务职责、具备复用价值） |
| **系统与环境配置**| `db_config.json`, `.env` | ✅ **长期保留**（维护连接与参数） |
| **标准工程文档** | `data_pipeline_spec.md`, `README.md` | ✅ **长期保留**（技术沉淀与移交） |
| **单次探索脚本** | `explore.py`, `check_views.py` | ❌ **强制清除**（结论得出后即无价值） |
| **单次探查脚本** | `inspect_*.py`, `search_*.py` | ❌ **强制清除**（零容忍残留） |
| **中间测试脚本** | `test_curve_decoding.py` | ❌ **强制清除**（转入正式单测或删除） |

---

## 四、检查与执行机制
1. **交卷前自检**：Agent 或开发者在宣布任务完成前，必须执行目录列表检查（`Get-ChildItem` / `ls`），确认没有散落的杂乱文件。
2. **违规即清**：任何时候发现根目录下存在未登记的临时脚本，一律无条件清理。
