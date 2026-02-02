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

## TP1：搜索算法详解 (基于 Notebook)

对应文件：[TP1-RechercheArborescente.ipynb](file:///d:/Desk/Daily/EDU%20TLS/M1/AI/TP_AI_miage/TP1/TP1-RechercheArborescente.ipynb)

本部分文档仅描述 Notebook 中实现的通用图搜索算法（Exercise 3 和 Exercise 5）。该框架使用 `Node` 类构建搜索树，并通过 `frontier` (边界) 的不同管理方式实现 BFS 和 DFS。

### 1. 核心数据结构

#### Node 类 (搜索树节点)
- **定义**：`Node(state, parent=None, action=None, path_cost=0)`
- **属性**：
  - `state`: 当前问题的状态（例如 Taquin 游戏的 3x3 矩阵）。
  - `parent`: 指向父节点的引用，用于回溯生成路径。
  - `action`: 从父节点到达当前节点所执行的动作（如 'up', 'down'）。
  - `path_cost`: 从根节点到当前节点的累积代价。
- **关键方法**：
  - `expand(problem)`: 生成当前节点的所有合法子节点列表。
  - `solution()`: 返回从根节点到当前节点的动作序列（解的路径）。

#### Problem 接口 (Taquin)
- `initial_state`: 初始状态。
- `is_goal(state)`: 判断是否为目标状态。
- `actions(state)`: 返回当前状态下的所有可用动作。
- `result(state, action)`: 返回执行动作后的新状态。

---

### 2. 算法实现逻辑详解

#### BFS (广度优先搜索) - Exercise 3
> **核心思想**：使用 FIFO 队列，先探索离根节点最近的层级。
> **Goal Test**：Early Goal Test (生成子节点时检查)。

- **数据结构**：
  - `frontier`: **列表** (模拟 FIFO 队列)，使用 `pop(0)` 取出首元素。
  - `explored`: **列表**，存储已扩展过的状态 (state)，避免重复搜索。

- **算法流程**：
  1. **初始化**：
     - 创建根节点 `node = Node(problem.initial_state)`。
     - 若 `problem.is_goal(node.state)`，直接返回结果。
     - 初始化 `frontier = [node]`，`explored = []`。
  2. **循环** (当 `frontier` 不为空时)：
     - **出队**：`node = frontier.pop(0)` (取出最早加入的节点)。
     - **记录**：将 `node.state` 加入 `explored`。
     - **扩展**：遍历 `node.expand(problem)` 生成的所有 `child` 节点。
     - **子节点处理**：
       - 若 `child.state` **不在** `explored` 且 **不在** `frontier` 中：
         - **目标检测**：若 `problem.is_goal(child.state)`，立即返回成功。
         - **入队**：`frontier.append(child)` (加入队尾)。
  3. **失败**：若队列空仍未找到解，返回 `None`。

#### DFS (深度优先搜索) - Exercise 5
> **核心思想**：使用 LIFO 栈，尽可能深地搜索树的分支。
> **Goal Test**：Late Goal Test (节点出栈时检查)。

- **数据结构**：
  - `frontier`: **列表** (模拟 LIFO 栈)，使用 `pop()` 取出尾元素。
  - `explored`: **列表**，存储已扩展过的状态。

- **算法流程**：
  1. **初始化**：
     - 创建根节点 `start_node`。
     - 初始化 `frontier = [start_node]`，`explored = []`。
  2. **循环** (当 `frontier` 不为空时)：
     - **出栈**：`node = frontier.pop()` (取出最后加入的节点)。
     - **目标检测**：若 `problem.is_goal(node.state)`，返回成功 (注意检测时机与 BFS 不同)。
     - **记录**：将 `node.state` 加入 `explored`。
     - **扩展**：获取所有子节点 `children = node.expand(problem)`。
     - **子节点处理**：
       - 遍历每个 `child`：
         - 若 `child.state` **不在** `explored` (且通常检查不在 `frontier`)：
           - **入栈**：`frontier.append(child)`。
  3. **失败**：若栈空仍未找到解，返回 `None`。

---

### 3. 运行方式

请使用 VS Code 或 Jupyter Notebook 打开并运行 [TP1-RechercheArborescente.ipynb](file:///d:/Desk/Daily/EDU%20TLS/M1/AI/TP_AI_miage/TP1/TP1-RechercheArborescente.ipynb)。
- 运行 **Exercise 3** 单元格以测试 BFS。
- 运行 **Exercise 5** 单元格以测试 DFS。

