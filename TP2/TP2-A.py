import heapq
from TP1.Node import Node  # 保持你的引用

def A_Star(problem, budget):
    """
    problem: 包含 initial_state, is_goal, 和 heuristic 方法
    budget: 最大探索节点数
    """
    
    # 1. 初始化起点
    start_node = Node(problem.initial_state)
    start_node.g = 0  # g: 已经走的步数
    # 假设 problem 里面有一个 heuristic(state) 方法计算 h
    start_node.h = problem.heuristic(start_node.state) 
    start_node.f = start_node.g + start_node.h
    
    # 2. 初始化 Frontier (使用最小堆)
    # 堆里的元素建议是元组: (f值, 节点对象)
    # 注意：Node 类最好实现了 __lt__ 方法以便比较，如果没有，可能会报错
    # 为了安全，也可以存 (f, id(node), node) 来避免直接比较 node
    frontier = [] 
    heapq.heappush(frontier, (start_node.f, start_node))
    
    # 用来记录已探索的状态，防止走回头路
    explored = set()
    
    count = 0

    while frontier:
        if count >= budget:
            return None, count

        # 3. 取出 f 最小的节点 (O(log N))
        # pop 出来的是元组 (f, node)，我们只需要 node
        current_f, node = heapq.heappop(frontier)

        # 4. 延迟检测 (Lazy Deletion)
        # 如果这个状态已经在 explored 里，说明我们要么处理过，
        # 要么之前通过更短的路径到达过它，直接跳过。
        if node.state in explored:
            continue
            
        # 5. 目标检测 (必须在 POP 出来时做)
        if problem.is_goal(node.state):
            return node, count
        
        # 加入已探索集合
        explored.add(node.state)
        count += 1

        # 6. 扩展邻居
        for child in node.expand(problem):
            # 如果孩子已经在 explored 中，通常跳过 (简单图搜索)
            if child.state in explored:
                continue
            
            # 计算 A* 的核心数值
            child.g = node.g + 1  # 假设每一步代价是 1
            child.h = problem.heuristic(child.state)
            child.f = child.g + child.h
            
            # 7. 加入堆 (Push)
            # 这里我们不检查 frontier 里是否有重复状态 (懒惰法)
            # 直接加进去，反正如果以后取出来发现重复了，上面第4步会过滤掉
            heapq.heappush(frontier, (child.f, child))

    return None, count