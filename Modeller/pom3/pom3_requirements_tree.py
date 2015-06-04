"""##############################################################################
   ### -@author: Joseph Krall
   ### -@note: POM3 Requirements Tree Module
   ### -@note: This work done in affiliation with the West Virginia University
   ### -@contact: kralljoe@gmail.com
   ### -@copyright: This work is made for academic research.  Do not edit without
   ### -@copyright: citing POM3 as a reference.
   ##############################################################################"""

class Requirement(object):

    def __init__(self, c, v):
        #cost of the task
        self.cost = c
        
        #value of the task (priority)
        self.value = v
        
        #if the task is done or not
        self.done = False
        
        #if the task is visible (invisible tasks technically have not been planned at all by project leaders)
        self.visible = False
    
    def __repr__(self): return "Done? " + str(self.done) + ".  (Cost: " + str(self.cost) + ", Value: " + str(self.value) + ")"


class requirements_node(object):
    def __init__(self, v, k, p=None, c=[], l=0):
        self.val = v
        self.key = k
        self.parent = p
        self.children = c
        self.level = l
        self.dependencies = []
        
    def add_child(self, v, k, l):
        self.children.append(requirements_node(v, k, self, [], l))
        
    def __repr__(self):
        if self.parent:
            return (" "*self.level + "{Key: " + self.key + ", Val: " + str(self.val) + ", ParentKey = " + self.parent.key + " Dep: " + str(self.dependencies) + "}")
        else: 
            return (" "*self.level + "{Key: " + self.key + ", Val: " + str(self.val) + ", ParentKey = None" + " Dep: " + str(self.dependencies) + "}")

    def show(self):
        print(self)
        for child in self.children:
            child.show()
            
    def iterative_search(self, k):
        for child in self.children:
            if child.key == k: return child
            else:
                result = child.iterative_search(k)
                if result: return result
                
    def max_depth(self):
        max = 0
        for child in self.children:
            if child.level > max: max = child.level
            m = child.max_depth()
            if m > max: max = m
        return max
    
    def traverse(self):
        list = []
        for child in self.children:
            list.append(child)
            to_add = child.traverse()
            for i in to_add:
                list.append(i)
        return list
            
            
        
class requirements_tree(object):
    def __init__(self):
        self.tree = []
    def add_root(self, v, k):
        self.tree.append(requirements_node(v, k, None, [], 0))
    def get_root(self, k):
        for root in self.tree:
            if root.key == k: return root
    def find_node(self, k):
        for root in self.tree:
            if root.key == k: return root
            else:
                result = root.iterative_search(k)
                if result: return result
    
    def get_level(self, root, level):
        list = []
        if (root.level < level):
            for child in root.children:
                list.append(self.get_level(child, level))
        elif (root.level == level):
            list.append(root.key)
        return list
    
    def traverse(self):
        list = []
        for root in self.tree:
            list.append(root)
            to_add = root.traverse()
            for i in to_add:
                list.append(i)
        return list
    
    def show(self):
        for root in self.tree:
            root.show()
