# Workspace File Cleanliness Rule

## Mandatory Instruction
1. DO NOT clutter the workspace root with temporary exploration or inspection scripts (e.g., `inspect_*.py`, `explore_*.py`, `test_*.py`, `check_*.py`, `tmp_*`).
2. When performing exploratory queries or debugging:
   - Prefer running code in-memory via `python -c "..."`.
   - If a file must be written, place it in a temporary subdirectory (e.g. `.temp/` or artifact scratch directory).
3. Upon completing the investigation or task, ALWAYS delete all temporary/scratch files immediately before concluding the turn.
4. Keep only long-term valuable files: core modules, configuration files, and standard documentation.
