
base_res = "new440x440-ep15.csv"

idx_12 = open("12" + base_res, "w")
idx_23 = open("23" + base_res, "w")
idx_13 = open("13" + base_res, "w")

with open(base_res) as fp:

    # read the first line of heads
    line = fp.readline();
    line = fp.readline();
    while line:
        l_array = line.rstrip().split(",")
        print(l_array[1].split(" "))
        incorrect = l_array[1].split(" ")[2]
        if (incorrect == "0"):
            idx_23.write(l_array[0] + "\n")
        if (incorrect == "1"):
            idx_13.write(l_array[0] + "\n")
        if (incorrect == "2"):
            idx_12.write(l_array[0] + "\n")


        line = fp.readline();
idx_12.close()
idx_23.close()
idx_13.close()


