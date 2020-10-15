
class test():
    def __init__(self,lists):
        self.list = lists

    def does(self):
        self.list.append('2')

global_list = []

cls_test = test(global_list)

for _ in range(10):
    cls_test.does()

print('cls_test.list',cls_test.list)
print('global_list',global_list)


import numpy as np

class test2():
    lists = np.zeros(10)
    def __init__(self):
        print('init',self.lists)

    def hstack(self,i):
        self.lists = np.hstack((self.lists[1:], i))
        print(self.lists)
    
test2 = test2()

for i in range(10):
    test2.hstack(i)
