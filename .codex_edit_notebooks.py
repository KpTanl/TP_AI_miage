import json
from pathlib import Path


def set_code_cell(nb, idx, text):
    if not text.endswith('\n'):
        text += '\n'
    nb['cells'][idx]['source'] = text.splitlines(keepends=True)

# ---------------------------
# TP1 notebook
# ---------------------------
path_tp1 = Path('TP1/TP1_Recherche_Arborescente_Non_informee.ipynb')
nb1 = json.loads(path_tp1.read_text(encoding='utf-8'))

set_code_cell(nb1, 17, '''def BFS(problem, budget=float('inf')):
    # 广度优先搜索（BFS）
    # - frontier 使用队列（pop(0)）
    # - count 记录已扩展节点数，用于 budget 截断
    node = Node(problem.initial_state)
    if problem.is_goal(node.state):
        return node, 0

    frontier = [node]  # FIFO 队列
    explored = []      # 已扩展状态
    count = 0

    while frontier:
        if count >= budget:
            return None

        node = frontier.pop(0)  # 取出最早入队节点
        explored.append(node.state)
        count += 1

        for child in node.expand(problem):
            if child.state in explored:
                continue

            # 避免把 frontier 中已有状态重复加入
            is_in_frontier = False
            for n in frontier:
                if n.state == child.state:
                    is_in_frontier = True
                    break

            if is_in_frontier:
                continue

            # Early Goal Test：生成子节点时立即检查目标
            if problem.is_goal(child.state):
                return child, count

            frontier.append(child)

    return None
''')

set_code_cell(nb1, 19, '''goal_state = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

def solve_and_display(initial_state, title):
    # BFS 调用示例：求解 + 输出路径 + 可视化
    print(f"{title}-----------:")

    # 1) 构造问题
    problem = Taquin(initial_state, goal_state, size=3)

    # 2) 调用 BFS
    result = BFS(problem)

    # 3) 解析并展示结果
    if result:
        solution_node, explored_count = result
        print(f"1. Nodes explored: {explored_count}")

        actions = solution_node.solution()
        print(f"2. Actions: {actions}")
        print(f"   Cost: {len(actions)}")

        print("3. Visualisation:")
        path_nodes = solution_node.path()

        for i, node in enumerate(path_nodes):
            print(f"Step {i}: {node.action if node.action else 'Start'}")
            visualize_state(node.state)
    else:
        print("No solution found")

# 调用方式：依次测试两个初始状态
state_1 = [
    [1, 2, 3],
    [4, 5, 6],
    [0, 7, 8]
]
solve_and_display(state_1, "Case 1")

state_2 = [
    [1, 2, 3],
    [4, 5, 0],
    [6, 7, 8]
]
solve_and_display(state_2, "Case 2")
''')

set_code_cell(nb1, 22, '''def DFS(problem, budget=float('inf')):
    # 深度优先搜索（DFS）
    # - frontier 使用栈（pop()）
    # - Goal Test 在节点弹出后执行（Late Goal Test）
    start_node = Node(problem.initial_state)
    frontier = [start_node]  # LIFO 栈
    explored = []
    count = 0

    while frontier:
        if count >= budget:
            return None

        node = frontier.pop()  # 取出最近入栈节点

        if problem.is_goal(node.state):
            return node, count

        explored.append(node.state)
        count += 1

        children = node.expand(problem)

        for child in children:
            if child.state in explored:
                continue

            # 避免 frontier 中重复状态
            is_in_frontier = False
            for n in frontier:
                if n.state == child.state:
                    is_in_frontier = True
                    break

            if not is_in_frontier:
                frontier.append(child)

    return None
''')

set_code_cell(nb1, 24, '''goal_state = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

def solve_and_display(initial_state, title):
    # DFS 调用示例：求解 + 输出路径 + 可视化
    print(f"{title}-----------:")

    # 1) 构造问题
    problem = Taquin(initial_state, goal_state, size=3)

    # 2) 调用 DFS（可设置 budget 防止搜索过深）
    result = DFS(problem, budget=10000)

    # 3) 解析并展示结果
    if result:
        solution_node, explored_count = result
        print(f"1. Nodes explored: {explored_count}")

        actions = solution_node.solution()
        print(f"2. Actions: {actions}")
        print(f"   Cost: {len(actions)}")

        print("3. Visualisation:")
        path_nodes = solution_node.path()

        for i, node in enumerate(path_nodes):
            print(f"Step {i}: {node.action if node.action else 'Start'}")
            visualize_state(node.state)
    else:
        print("No solution found")

# 调用方式：依次测试两个初始状态
state_1 = [
    [1, 2, 3],
    [4, 5, 6],
    [0, 7, 8]
]
solve_and_display(state_1, "Case 1")

state_2 = [
    [1, 2, 3],
    [4, 5, 0],
    [6, 7, 8]
]
solve_and_display(state_2, "Case 2")
''')

path_tp1.write_text(json.dumps(nb1, ensure_ascii=False, indent=1), encoding='utf-8')

# ---------------------------
# TP2 notebook
# ---------------------------
path_tp2 = Path('TP2/TP_Recherche_Arborescente_Informee.ipynb')
nb2 = json.loads(path_tp2.read_text(encoding='utf-8'))

set_code_cell(nb2, 6, '''goal_state = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]
def calcule_h1(state, goal_state):
    # H1：统计“错位数字块”数量（忽略空格 0）
    count = 0
    for i in range(len(goal_state)):
        for j in range(len(goal_state[i])):
            if state[i][j] != 0 and state[i][j] != goal_state[i][j]:
                count += 1
    return count
''')

set_code_cell(nb2, 11, '''def calcule_h2(problem, goal_state):
    # H2：曼哈顿距离（每个数字到目标位置的行列距离之和）
    state = problem
    dist = 0

    for r, row in enumerate(state):
        for c, tile in enumerate(row):
            if tile == 0:
                continue

            # 在目标状态里找到 tile 的目标坐标 (gr, gc)
            gr, gc = next(
                (i, j)
                for i, grow in enumerate(goal_state)
                for j, val in enumerate(grow)
                if val == tile
            )

            # 曼哈顿距离：|r-gr| + |c-gc|
            dist += abs(r - gr) + abs(c - gc)

    return dist
''')

set_code_cell(nb2, 16, '''def A_etoile(problem, heuristique, budget=float('inf')):
    # A* 搜索实现
    # - 评价函数：f(n) = g(n) + h(n)
    # - frontier 中保存 (node, f)
    # - 每轮扩展 f 最小的节点
    node = Node(problem.initial_state)

    # 起点的 f 值
    h_val = heuristique(node.state, problem.goal_state)
    f_val = node.path_cost + h_val

    frontier = [(node, f_val)]
    explored = []
    count = 0

    while frontier and count < budget:
        # 1) 找出 frontier 里 f 最小的节点
        min_idx = 0
        for i in range(1, len(frontier)):
            if frontier[i][1] < frontier[min_idx][1]:
                min_idx = i

        # 2) 取出该节点并计数
        node, f_val = frontier.pop(min_idx)
        count += 1

        # 3) 目标检测
        if problem.is_goal(node.state):
            return node, count

        explored.append(node.state)

        # 4) 扩展子节点
        for child in node.expand(problem):
            if child.state in explored:
                continue

            h_child = heuristique(child.state, problem.goal_state)
            f_child = child.path_cost + h_child

            # 5) 若子状态已在 frontier，仅保留更优 f
            in_frontier = False
            for i in range(len(frontier)):
                existing_node, existing_f = frontier[i]
                if existing_node.state == child.state:
                    in_frontier = True
                    if f_child < existing_f:
                        frontier[i] = (child, f_child)
                    break

            # 6) 新状态直接入 frontier
            if not in_frontier:
                frontier.append((child, f_child))

    return None, count

"""
不考虑真实步数 g(n)，只按 h(n) 选点就是贪婪最佳优先搜索（Greedy Best-First）。
如果希望自定义“先比 h2，再比 h1”，可以把优先级写成元组 (h2, h1)。
"""
''')

set_code_cell(nb2, 18, '''initial_state_ex6 = [
    [1, 2, 3],
    [4, 5, 6],
    [0, 7, 8]
]

goal_state = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

problem_ex6 = Taquin(initial_state_ex6, goal_state, 3)

# 调用方式 1：A* + H1（错位块）
print("=== Résolution avec A* et H1 (Tuiles mal placées) ===")
node_h1, count_h1 = A_etoile(problem_ex6, calcule_h1)

if node_h1:
    print(f"Nombre de nœuds explorés : {count_h1}")
    print(f"Solution (Chemin) : {node_h1.solution()}")
    print(f"Coût du chemin : {node_h1.path_cost}")
else:
    print("Pas de solution trouvée.")

print("\n" + "="*50 + "\n")

# 调用方式 2：A* + H2（曼哈顿距离）
print("=== Résolution avec A* et H2 (Distance de Manhattan) ===")
node_h2, count_h2 = A_etoile(problem_ex6, calcule_h2)

if node_h2:
    print(f"Nombre de nœuds explorés : {count_h2}")
    print(f"Solution (Chemin) : {node_h2.solution()}")
    print(f"Coût du chemin : {node_h2.path_cost}")
else:
    print("Pas de solution trouvée.")
''')

path_tp2.write_text(json.dumps(nb2, ensure_ascii=False, indent=1), encoding='utf-8')

# ---------------------------
# TP3 notebook
# ---------------------------
path_tp3 = Path('TP3/TP_SAT_et_CSP.ipynb')
nb3 = json.loads(path_tp3.read_text(encoding='utf-8'))

set_code_cell(nb3, 10, '''from pysat.formula import CNF

class Sudoku_SAT:
    # 将 9x9 Sudoku 编码为 SAT（CNF）问题
    # var_map[(i, j, k)] = 变量编号，表示“第 i 行第 j 列取值 k”
    def __init__(self, grid):
        self.grid = grid
        self.cnf = CNF()  # CNF 子句容器
        self.var_map = {}  # (行, 列, 值) -> SAT 变量 ID

        # 1) 创建 SAT 变量映射
        var_count = 1
        for i in range(9):
            for j in range(9):
                for k in range(1, 10):
                    self.var_map[(i, j, k)] = var_count
                    var_count += 1

        # 2) 生成 CNF 约束子句

        # a) 每个格子至少有一个值
        for i in range(9):
            for j in range(9):
                clause = [self.var_map[(i, j, k)] for k in range(1, 10)]
                self.cnf.append(clause)

        # b) 每个格子至多有一个值
        for i in range(9):
            for j in range(9):
                for k1 in range(1, 10):
                    for k2 in range(k1 + 1, 10):
                        clause = [-self.var_map[(i, j, k1)], -self.var_map[(i, j, k2)]]
                        self.cnf.append(clause)

        # c) 每一行必须包含 1..9（至少一次）
        for i in range(9):
            for k in range(1, 10):
                clause = [self.var_map[(i, j, k)] for j in range(9)]
                self.cnf.append(clause)

        # d) 每一列必须包含 1..9（至少一次）
        for j in range(9):
            for k in range(1, 10):
                clause = [self.var_map[(i, j, k)] for i in range(9)]
                self.cnf.append(clause)

        # e) 每个 3x3 宫必须包含 1..9（至少一次）
        for subgrid_row in range(3):
            for subgrid_col in range(3):
                for k in range(1, 10):
                    clause = [
                        self.var_map[(subgrid_row * 3 + i, subgrid_col * 3 + j, k)]
                        for i in range(3)
                        for j in range(3)
                    ]
                    self.cnf.append(clause)

        # f) 初始已知数字约束（单位子句）
        for i in range(9):
            for j in range(9):
                if self.grid[i][j] != 0:
                    self.cnf.append([self.var_map[(i, j, self.grid[i][j])]])
''')

set_code_cell(nb3, 14, '''# SAT 调用示例：先定义待求解网格，再调用 solve_sudoku_sat
grid = [
    [0, 0, 0, 7, 0, 0, 0, 0, 0],
    [0, 0, 7, 0, 5, 1, 0, 0, 0],
    [9, 0, 5, 0, 3, 8, 0, 0, 7],
    [0, 9, 0, 5, 0, 7, 0, 1, 0],
    [4, 0, 1, 0, 0, 0, 3, 6, 0],
    [2, 0, 0, 1, 0, 0, 0, 7, 0],
    [0, 0, 4, 0, 8, 0, 0, 0, 3],
    [0, 3, 0, 0, 7, 0, 0, 0, 2],
    [0, 0, 6, 0, 0, 0, 0, 0, 0],
]

res = solve_sudoku_sat(grid)
if res:
    visualize_sudoku(res)
else:
    print("aucune solution")
''')

set_code_cell(nb3, 17, '''from constraint import Problem, AllDifferentConstraint

def solve_sudoku_csp_no_sugar(grid):
    # 1) 创建 CSP 问题实例
    problem = Problem()

    # ==========================================
    # 第一步：定义变量 (Variables) 和值域 (Domains)
    # ==========================================
    for i in range(9):      # 行索引
        for j in range(9):  # 列索引
            current_val = grid[i][j]

            # 空格取值域 1..9；已知数字则值域固定为 [该数字]
            domain = []
            if current_val == 0:
                for num in range(1, 10):
                    domain.append(num)
            else:
                domain.append(current_val)

            # 变量用坐标元组表示，例如 (0, 0)
            problem.addVariable((i, j), domain)

    # ==========================================
    # 第二步：添加约束 (Constraints)
    # 核心规则：AllDifferentConstraint
    # ==========================================

    # 2.1 行约束
    for i in range(9):
        row_cells = []
        for j in range(9):
            row_cells.append((i, j))
        problem.addConstraint(AllDifferentConstraint(), row_cells)

    # 2.2 列约束
    for j in range(9):
        col_cells = []
        for i in range(9):
            col_cells.append((i, j))
        problem.addConstraint(AllDifferentConstraint(), col_cells)

    # 2.3 3x3 宫约束
    for br in [0, 3, 6]:
        for bc in [0, 3, 6]:
            block_cells = []
            for di in range(3):
                for dj in range(3):
                    row = br + di
                    col = bc + dj
                    block_cells.append((row, col))
            problem.addConstraint(AllDifferentConstraint(), block_cells)

    # ==========================================
    # 第三步：求解并格式化输出
    # ==========================================
    solutions = problem.getSolutions()

    if len(solutions) == 0:
        print("Aucune solution n'a été trouvée.")
        return []

    formatted_grids = []
    for sol in solutions:
        solved_grid = []
        for _ in range(9):
            row = [0] * 9
            solved_grid.append(row)

        for coordinate, value in sol.items():
            row_idx = coordinate[0]
            col_idx = coordinate[1]
            solved_grid[row_idx][col_idx] = value

        formatted_grids.append(solved_grid)

    return formatted_grids


def solve_sudoku_csp(grid):
    # 兼容调用名：返回候选解列表（通常取 res[0] 展示）
    return solve_sudoku_csp_no_sugar(grid)
''')

set_code_cell(nb3, 19, '''# CSP 调用示例：0 表示空格
grid = [[5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]]
''')

set_code_cell(nb3, 20, '''# 调用方式：solve_sudoku_csp 返回解列表，通常取第 1 个解展示
res = solve_sudoku_csp(grid)
if res:
    visualize_sudoku(res[0])
else:
    print("aucune solution")
''')

path_tp3.write_text(json.dumps(nb3, ensure_ascii=False, indent=1), encoding='utf-8')

# ---------------------------
# README append TP3 SAT/CSP section
# ---------------------------
readme_path = Path('README.md')
readme = readme_path.read_text(encoding='utf-8')
marker = '## TP3：SAT 与 CSP（TP-SAT）'
if marker not in readme:
    add = '''

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
'''
    readme = readme + add
    readme_path.write_text(readme, encoding='utf-8')

print('Notebook and README updates completed.')
