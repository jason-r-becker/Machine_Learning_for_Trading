import numpy as np


def printmap(data):
    print "--------------------"
    for row in range(0, data.shape[0]):
        for col in range(0, data.shape[1]):
            if data[row,col] == 0: # Empty space
                print " ",
            if data[row,col] == 1: # Obstacle
                print "O",
            if data[row,col] == 2: # El roboto
                print "*",
            if data[row,col] == 3: # Goal
                print "X",
            if data[row,col] == 4: # Trail
                print ".",
            if data[row,col] == 5: # Quick sand
                print "~",
            if data[row,col] == 6: # Stepped in quicksand
                print "@",
        print
    print "--------------------"
for i in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']:
    filename = 'testworlds/world' + i + '.csv'
    inf = open(filename)
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])
    print('World: ' + i)
    printmap(data)
