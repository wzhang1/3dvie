labels = {}
with open("../data/image_dev.txt") as fp:
    line = fp.readline();
    line = fp.readline();
    while line:
        l_array = line.rstrip().split(",")
            #print(l_array)
        labels[l_array[0]] = int(l_array[1])

        line = fp.readline();

#print(labels)
correct = 0

wrong = 0
# be careful, some of the model are traine with background classes


with open("../sub1/new500x500-ep11.csv") as fp:

    line = fp.readline();
    line = fp.readline();
    while line:
        l_array = line.rstrip().split(",")
        #print(labels[l_array[0]])
        #print(l_array)
        predict1 = int(l_array[1][0])
        #print(predict)
        predict2 = int(l_array[1][2])
        #print(predict2)
        
        if (predict2 == labels[l_array[0]] or predict1 == labels[l_array[0]]):
            correct = correct + 1
        else:
            wrong = wrong + 1

        line = fp.readline();
'''
#with open("./new600x600-ep10.csv") as fp:
#with open("./12Base3_classes300x300-ep15.csv") as fp:
with open("./13new440x440-ep15.csv") as fp:

    line = fp.readline();
    while line:
        l_array = line.rstrip().split(",")
        print(labels[l_array[0]])
        if (1 == labels[l_array[0]] or 3 == labels[l_array[0]]):
            correct = correct + 1
        else:
            wrong = wrong + 1

        line = fp.readline();


with open("./23new440x440-ep15.csv") as fp:

    line = fp.readline();
    while line:
        l_array = line.rstrip().split(",")
        print(labels[l_array[0]])
        if (2 == labels[l_array[0]] or 3 == labels[l_array[0]]):
            correct = correct + 1
        else:
            wrong = wrong + 1

        line = fp.readline();

'''
print(correct)
print(wrong)
print(correct * 1.0 /(correct + wrong))

