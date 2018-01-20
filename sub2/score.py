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
#with open("./new600x600-ep10.csv") as fp:
#with open("./12Base3_classes300x300-ep15.csv") as fp:
#with open("./Base3_classes300x300-ep15.csv12.csv") as fp:
with open("./partial12.csv") as fp:

    line = fp.readline();
    line = fp.readline();
    while line:
        l_array = line.rstrip().split(",")
            #print(l_array)
        res = l_array[1].split(" ")[0]
        if (int(res) + 1 == labels[l_array[0][12:]]):
            correct = correct + 1
        else:
            wrong = wrong + 1

        line = fp.readline();


print(correct)
print(wrong)
print(correct * 1.0 /(correct + wrong))

