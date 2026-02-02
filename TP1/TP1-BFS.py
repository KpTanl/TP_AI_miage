from TP1.Node import Node
from collections import deque
def BFS(problem,budget):
    """
    problems: liste de problèmes
    budget: nombre maximum d'états à explorer
    """
    node = Node(problem.initial_state) #初始化

    if problem.is_goal(node.state): #如果发现找到答案
        return node,0

    frontier = deque([node]) #将要处理的顺序
    frontier_set = {node.state} # 加速寻找是否在frontier里。 已发现，未处理
    explored = set() # 已处理
    count = 0

    while frontier :
        if count>=budget:
            return None
        
        node = frontier.popleft()
        frontier_set.remove(node.state)
        explored.add(node.state)
        count += 1

        for child in node.expand(problem):
            #O(1)
            if child.state in explored or child.state in frontier_set:  
                continue
            if problem.is_goal(child.state):
                return child,count
            
            frontier.append(child)
            frontier_set.add(child.state)
    return None