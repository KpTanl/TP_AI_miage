# AI - TP（Toulouse EDU / M1）

本仓库用于存放课程/作业的 TP 代码与材料。

## 目录结构

- `TP1/`：搜索相关练习（包含 n 皇后 BFS）
- `debut/`：Python 基础与 Pandas 等入门练习（Notebook、数据集、题目）
- `dataRessource/`：课程用数据资源

## 环境（Environment）

- 操作系统：Windows（本地路径示例为 Windows）
- Python：3.12（或 3.x）
- 依赖：仅使用 Python 标准库（当前代码无需额外安装第三方包）

可选：使用虚拟环境（推荐）

```bat
python -m venv .venv
.\.venv\Scripts\activate
python -V
```

## TP1：n 皇后（Dames.py）

### 算法说明

- 使用 BFS（宽度优先搜索）生成部分解
- 解的表示：列表 `table = [c0, c1, ..., c_{n-1}]`
  - 索引 = 行号（row）
  - 元素 = 列号（col）
- 冲突规则：同列冲突 / 同对角线冲突（行差绝对值等于列差绝对值）

### 运行方式

在仓库根目录执行：

```bat
python .\TP1\Dames.py
```

程序会打印一个解（`ans`）以及类内部保存的解（`table`）。

