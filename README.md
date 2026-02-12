# AI - TP（Toulouse EDU / M1）

本仓库用于存放课程/作业的 TP 代码与材料。

## 环境（Environment）

- 操作系统：Windows（本地路径示例为 Windows）
- Python：3.12（或 3.x）
- 依赖：仅使用 Python 标准库（当前代码无需额外安装第三方包）


## TP1/TP2：搜索算法详解（基于 Notebook）

主要参考（考试复习以这两个为准）：
- TP1（非信息搜索，BFS/DFS）：[TP1_Recherche_Arborescente_Non_informee.ipynb](file:///d:/Desk/Daily/EDU%20TLS/M1/AI/TP_AI_miage/TP1/TP1_Recherche_Arborescente_Non_informee.ipynb)
- TP2（信息搜索，Greedy/A*）：[TP_Recherche_Arborescente_Informee.ipynb](file:///d:/Desk/Daily/EDU%20TLS/M1/AI/TP_AI_miage/TP2/TP_Recherche_Arborescente_Informee.ipynb)

本部分文档以这两个 Notebook 的实现为主线：TP1 通过 `frontier`（边界）的不同管理方式实现 BFS/DFS；TP2 在此基础上引入启发式函数，并实现 Greedy Best-First Search 与 A*（A_etoile）。

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

### 3. 复习：以 BFS 为基准的三算法对照（TP1/TP2 实现）

#### 3.1 一张表背下来：BFS / DFS / A*

| 编号 | BFS（TP1：BFS，代码片段） | DFS（TP1：DFS，代码片段） | A*（TP2：A_etoile，代码片段） |
| :-- | :-- | :-- | :-- |
| 1 | `frontier = [node]`<br>`explored = []` | `frontier = [start_node]`<br>`explored = []` | `frontier = [(node, f_val)]`<br>`explored = []` |
| 2 | `node = frontier.pop(0)` | `node = frontier.pop()` | `min_idx = 0`<br>`for i in range(1, len(frontier)):`<br>`    if frontier[i][1] < frontier[min_idx][1]:`<br>`        min_idx = i`<br>`node, f_val = frontier.pop(min_idx)` |
| 3 | `if problem.is_goal(child.state):`<br>`    return child, count` | `if problem.is_goal(node.state):`<br>`    return node, count` | `if problem.is_goal(node.state):`<br>`    return node, count` |
| 4 | `if child.state in explored:`<br>`    continue`<br>`for n in frontier:`<br>`    if n.state == child.state:` | `if child.state in explored:`<br>`    continue`<br>`for n in frontier:`<br>`    if n.state == child.state:` | `if child.state in explored:`<br>`    continue`<br>`for i in range(len(frontier)):`<br>`    existing_node, existing_f = frontier[i]` |
| 5 | `for child in node.expand(problem):` | `children = node.expand(problem)`<br>`for child in children:` | `h_child = heuristique(child.state, problem.goal_state)`<br>`f_child = child.path_cost + h_child` |
| 6 | `frontier.append(child)` | `frontier.append(child)` | `if f_child < existing_f:`<br>`    frontier[i] = (child, f_child)`<br>`if not in_frontier:`<br>`    frontier.append((child, f_child))` |

---

#### 3.2 BFS 基准代码（TP1 Notebook 同款）

```python
def BFS(problem, budget=float("inf")):
    node = Node(problem.initial_state)
    if problem.is_goal(node.state):
        return node, 0

    frontier = [node]
    explored = []
    count = 0

    while frontier:
        if count >= budget:
            return None

        node = frontier.pop(0)
        explored.append(node.state)
        count += 1

        for child in node.expand(problem):
            if child.state in explored:
                continue

            is_in_frontier = False
            for n in frontier:
                if n.state == child.state:
                    is_in_frontier = True
                    break
            if is_in_frontier:
                continue

            if problem.is_goal(child.state):
                return child, count

            frontier.append(child)

    return None
```

---

#### 3.3 DFS：在 BFS 基准上改哪些地方（TP1 Notebook）

把 BFS 当“基准代码”，DFS 主要改三类点：
- 取节点从 `pop(0)` 变成 `pop()`（队列 -> 栈）
- Goal Test 从“生成子节点时”移到“弹出节点时”
- 子节点只负责入栈，不在循环内做 early goal test

```diff
- node = Node(problem.initial_state)
- if problem.is_goal(node.state):
-     return node, 0
- frontier = [node]
+ start_node = Node(problem.initial_state)
+ frontier = [start_node]

- node = frontier.pop(0)
+ node = frontier.pop()

+ if problem.is_goal(node.state):
+     return node, count

- if problem.is_goal(child.state):
-     return child, count
```

---

#### 3.4 A*：在 BFS 基准上“多出来的东西”（TP2 Notebook）

TP2 的 A*（函数名 `A_etoile`）的核心变化是：`frontier` 不再只存节点，而是存 `(node, f)`，并且每轮选择 `f` 最小的节点扩展。

把 BFS 当“基准代码”，A* 主要改这些地方（TP2 Notebook 的写法：列表 + 扫描最小 `f`）：

```diff
- def BFS(problem, budget=float("inf")):
+ def A_etoile(problem, heuristique, budget=float("inf")):
     node = Node(problem.initial_state)
-    if problem.is_goal(node.state):
-        return node, 0
 
-    frontier = [node]
+    h_val = heuristique(node.state, problem.goal_state)
+    f_val = node.path_cost + h_val
+    frontier = [(node, f_val)]
     explored = []
     count = 0
 
-    while frontier:
+    while frontier and count < budget:
         if count >= budget:
             return None
 
-        node = frontier.pop(0)
-        explored.append(node.state)
-        count += 1
+        min_idx = 0
+        for i in range(1, len(frontier)):
+            if frontier[i][1] < frontier[min_idx][1]:
+                min_idx = i
+
+        node, f_val = frontier.pop(min_idx)
+        count += 1
+
+        if problem.is_goal(node.state):
+            return node, count
+
+        explored.append(node.state)
 
         for child in node.expand(problem):
             if child.state in explored:
                 continue
 
-            is_in_frontier = False
-            for n in frontier:
-                if n.state == child.state:
-                    is_in_frontier = True
-                    break
-            if is_in_frontier:
-                continue
-
-            if problem.is_goal(child.state):
-                return child, count
-
-            frontier.append(child)
+            h_child = heuristique(child.state, problem.goal_state)
+            f_child = child.path_cost + h_child
+
+            in_frontier = False
+            for i in range(len(frontier)):
+                existing_node, existing_f = frontier[i]
+                if existing_node.state == child.state:
+                    in_frontier = True
+                    if f_child < existing_f:
+                        frontier[i] = (child, f_child)
+                    break
+
+            if not in_frontier:
+                frontier.append((child, f_child))
 
-    return None
+    return None, count
```


#### 3.5 记忆法（面向考试）

- BFS：按层推进，容器是队列，目标检测更早（生成子节点时就看）
- DFS：一路走到底，容器是栈，目标检测更晚（弹出节点时才看）
- A*：每步挑 “`g+h` 最小” 的节点扩展，`frontier` 里要带上 `f` 并支持更优路径覆盖

---

### 4. 运行方式

请使用 VS Code 或 Jupyter Notebook 打开并运行：
- TP1（非信息搜索）：[TP1_Recherche_Arborescente_Non_informee.ipynb](file:///d:/Desk/Daily/EDU%20TLS/M1/AI/TP_AI_miage/TP1/TP1_Recherche_Arborescente_Non_informee.ipynb)
  - 运行 BFS：`def BFS(problem, budget=float('inf'))`（Exercise 3）
  - 运行 DFS：`def DFS(problem, budget=float('inf'))`（Exercise 5）
- TP2（信息搜索）：[TP_Recherche_Arborescente_Informee.ipynb](file:///d:/Desk/Daily/EDU%20TLS/M1/AI/TP_AI_miage/TP2/TP_Recherche_Arborescente_Informee.ipynb)
  - 运行 A*：`def A_etoile(problem, heuristique, budget=float('inf'))`（创建函数处）
  - 对比两种启发式：Exercise 6（分别用 `calcule_h1` 与 `calcule_h2`）


## TP3：SAT 与 CSP（TP-SAT）

参考文件：
- [TP_SAT_et_CSP.ipynb](file:///d:/Desk/Daily/EDU%20TLS/M1/AI/TP_AI_miage/TP3/TP_SAT_et_CSP.ipynb)

### 1. 实现方式

#### 1.1 SAT 实现（`Sudoku_SAT` + `solve_sudoku_sat`）
- 使用 `pysat` 将 Sudoku 编码为 CNF：变量 `var_map[(i, j, k)]` 表示“第 i 行第 j 列是数字 k”。
- 约束包含：
  - 每格至少一个值。
  - 每格至多一个值。
  - 每行/每列/每个 3x3 宫都包含 1~9。
  - 初始已知数字用单位子句固定。
- 调用 `Glucose3` 求解并把 SAT 模型反解回 9x9 网格。

#### 1.2 CSP 实现（`solve_sudoku_csp_no_sugar` / `solve_sudoku_csp`）
- 使用 `python-constraint` 建模：
  - 变量：每个格子 `(i, j)`。
  - 值域：空格为 `1..9`，已知数字为单值域。
  - 约束：行/列/宫分别加 `AllDifferentConstraint()`。
- `solve_sudoku_csp` 返回候选解列表，通常取第一个解展示。

### 2. 调用方式（重点）

#### 2.1 SAT 调用
```python
res = solve_sudoku_sat(grid)
if res:
    visualize_sudoku(res)
```

#### 2.2 CSP 调用
```python
res = solve_sudoku_csp(grid)
if res:
    visualize_sudoku(res[0])
```

### 3. Notebook 运行顺序（TP-SAT）
1. 安装依赖（按需）：`python-sat`、`python-constraint`。
2. 运行 SAT 建模和求解单元：`Sudoku_SAT` -> `solve_sudoku_sat` -> SAT 示例调用。
3. 运行 CSP 建模和求解单元：`solve_sudoku_csp_no_sugar`/`solve_sudoku_csp` -> CSP 示例调用。
4. 最后运行比较单元：`comparer_solveurs` 与表格输出。
