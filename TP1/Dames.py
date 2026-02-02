"""八皇后问题：使用宽度优先搜索（BFS）寻找不互相攻击的 n 个皇后排布。
- 部分解以列表表示：索引为行号，元素为列号
- 冲突规则：同列或处于同一对角线（行差等于列差）
"""
from queue import Queue


class Dames:
    def __init__(self):
        self.table = []  # 存储一个完整解（或第一个解）的列位置列表，如 [c0, c1, ..., c_{n-1}]

    def verifier_dames(self, table, curr) -> bool:
        """在部分解 table 的下一行放置列 curr 是否安全
        安全规则：
        - 不同列：已有列 c 不能等于 curr
        - 不同对角线：行差等于列差表示在同一对角线，需避免
        参数：
        - table: 已放置皇后的列位置列表，索引为行号
        - curr: 计划在下一行放置的列
        返回：True 可放置；False 冲突
        """
        row = len(table)  # 下一行的行号
        for r, c in enumerate(table):  # 遍历已放置皇后 (行 r, 列 c)
            same_col = (c == curr)
            same_diag = (abs(row - r) == abs(curr - c))
            if same_col or same_diag:
                return False
        return True

        
    def bfs(self, n: int = 8, find_all: bool = False):
        """使用队列的宽度优先搜索（BFS）求解 n 皇后
        定义：
        - 部分解：长度为 k 的列表 [c0, c1, ..., c_{k-1}]，表示前 k 行的列选择
        步骤：
        1) 从空部分解 [] 开始入队
        2) 出队一个部分解，扩展到下一行的所有安全列
        3) 放满 n 行即得到一个完整解
        参数：
        - n: 棋盘大小（默认 8）
        - find_all: True 收集所有解；False 返回第一个解
        返回：
        - find_all=False: 一个完整解列表；找不到返回 []
        - find_all=True: 所有解的列表；可能为空
        同步：self.table 更新为找到的第一个解
        """
        q = Queue()                  # 维护待扩展的部分解（FIFO 队列）
        q.put([])                    # 初始为空部分解（尚未放置任何皇后）
        res = []                     # 收集全部解（仅在 find_all=True 时使用）
        while not q.empty():         # 层序展开：保证最短步数先到达完整解
            partial = q.get()        # 取出一个部分解
            row = len(partial)       # 下一步要扩展的行号
            if row == n:             # 已放置 n 行 -> 完整解
                if find_all:
                    res.append(partial)
                    continue
                self.table = partial 
                return partial
            for c in range(n):       # 尝试当前行的每一个列选择
                if self.verifier_dames(partial, c):  # 安全检查
                    q.put(partial + [c])             # 入队新的部分解
        if find_all:
            if res:
                self.table = res[0]
            return res
        return []

    def dfs(self, n: int = 8, find_all: bool = False):
        def search(partial):
            row = len(partial)
            if row == n:
                if find_all:
                    res.append(partial)
                    return None
                return partial
            for c in range(n):
                if self.verifier_dames(partial, c):
                    found = search(partial + [c])
                    if found is not None:
                        return found
            return None

        res = []
        found = search([])
        if find_all:
            if res:
                self.table = res[0]
            return res
        if found is None:
            return []
        self.table = found
        return found


if __name__ == "__main__":
    d = Dames()
    solution = d.bfs(n=8, find_all=False)
    print("ans", solution)
    print("table：", d.table)


