from pom3_requirements_tree import *
import random

"""##############################################################################
   ### -@author: Joseph Krall
   ### -@note: POM3 Requirements Heap Module
   ### -@note: This work done in affiliation with the West Virginia University
   ### -@contact: kralljoe@gmail.com
   ### -@copyright: This work is made for academic research.  Do not edit without
   ### -@copyright: citing POM3 as a reference.
   ##############################################################################"""

def random_cost(): return random.randint(1, 100)
def random_value(): return random.randint(1, 100)


class pom3_requirements:
    def __init__(requirements, decisions):
        requirements.heap = requirements_tree()
        requirements.count = int(2.5*[3,10,30,100,300][decisions.size])
        requirements.decisions = decisions
        
        for i in range(requirements.count):
            requirements.heap.add_root(Requirement((decisions.size+1)*random_cost(), random_value()), 'Base Req #' + str('%.3d'%(i+1)))
            
            parent = requirements.heap.tree[i]
            requirements.recursive_adder(parent, 1)
        
        
        #Add dependencies
        for i in range(requirements.count):
            rand = random.randint(1,1000)
            if (rand <= 15):
                #pick a requirement at this level, of the next base tree
                level = 0 
                if ((i+1) < len(requirements.heap.tree)): 
                    req_node = requirements.heap.tree[i+1]
                    adderDie = random.randint(1,100)
                    if adderDie <= decisions.interdependency: requirements.add_dependency(requirements.heap.tree[i], req_node)    
            requirements.recursive_dep_adder(requirements.heap.tree[i], i, 1)        
        #linearize the list
        requirements.tasks = requirements.heap.traverse()
        
        #TEsting
        #for i in requirements.tasks:
        #    print i

    def add_children(self, num, parent, level):
        for c in range(num):
            parent.add_child(Requirement(random_cost(), random_value()), "+"*level + 'Child-' + parent.key[0] + parent.key[len(parent.key)-3] + parent.key[len(parent.key)-2] + parent.key[len(parent.key)-1] + ' #' + str('%.3d'%(c+1)), level)
            self.recursive_adder(parent.children[c], level+1)
        
    def add_dependency(self, dep_node, req_node):
        #Add a dependency from this node to another node at the same level of the next root
        #We store the key of the requirement_node in the list of the dependent_node's dependencies
        dep_node.dependencies.append(req_node)
        
    def recursive_adder(self, parent, level):
        #Random exponential chance that we add child node:
        rand = random.randint(1,1000)
        odds = [15, 30, 60, 120, 240]
        
        for numChildren,chance in enumerate(odds): 
            if (rand <= chance): 
                self.add_children(5-numChildren, parent, level)
                break
                
    def recursive_dep_adder(self, parent, rootIndex, level):
        
        if (len(parent.children) > 0 and ((rootIndex+1) < len(self.heap.tree))):
            if (level <= self.heap.tree[rootIndex+1].max_depth()):
                rand = random.randint(1,1000)
                odds = [15, 30, 60, 120, 240, 500]
                
                if level > 5: oddsInd = 5 
                else: oddsInd = level
                
                if (rand <= odds[oddsInd]):
                    #pick a random child at this level of this root
                    rand = random.randint(0,len(parent.children)-1)
                    randChild= parent.children[rand]
                    
                    #pick a random node at the level of the next root
                    levelNodes = self.heap.get_level(self.heap.tree[rootIndex+1], level)
                    rand = random.randint(0, len(levelNodes)-1)
                    
                    #add the dependency from randChild to levelNodes[rand]
                    adderDie = random.randint(1,100)
                    if adderDie <= self.decisions.interdependency: self.add_dependency(randChild, levelNodes[rand])
                for child in parent.children:
                    self.recursive_dep_adder(child, rootIndex, level+1)
