from random import seed
from random import randint

zero = [0,1,1,1,0,
        1,0,0,0,1,
        1,0,0,0,1,
        1,0,0,0,1,
        1,0,0,0,1,
        1,0,0,0,1,
        1,0,0,0,1,
        1,0,0,0,1,
        0,1,1,1,0]

one =  [0,0,1,0,0,
        0,1,1,0,0,
        1,0,1,0,0,
        0,0,1,0,0,
        0,0,1,0,0,
        0,0,1,0,0,
        0,0,1,0,0,
        0,0,1,0,0,
        0,0,1,0,0]

two =  [0,1,1,1,0,
        1,0,0,0,1,
        0,0,0,0,1,
        0,0,0,0,1,
        0,0,0,1,0,
        0,0,1,0,0,
        0,1,0,0,0,
        1,0,0,0,0,
        1,1,1,1,1]

three =[0,1,1,1,0,
        1,0,0,0,1,
        0,0,0,0,1,
        0,0,0,0,1,
        0,0,0,1,0,
        0,0,0,0,1,
        0,0,0,0,1,
        1,0,0,0,1,
        0,1,1,1,0]

four = [0,0,0,1,0,
        0,0,1,1,0,
        0,0,1,1,0,
        0,1,0,1,0,
        0,1,0,1,0,
        1,0,0,1,0,
        1,1,1,1,1,
        0,0,0,1,0,
        0,0,0,1,0]

five = [1,1,1,1,1,
        1,0,0,0,0,
        1,0,0,0,0,
        1,1,1,1,0,
        1,0,0,0,1,
        0,0,0,0,1,
        0,0,0,0,1,
        1,0,0,0,1,
        0,1,1,1,0]

six = [0,1,1,1,0,
        1,0,0,0,1,
        1,0,0,0,0,
        1,0,0,0,0,
        1,1,1,1,0,
        1,0,0,0,1,
        1,0,0,0,1,
        1,0,0,0,1,
        0,1,1,1,0]

seven =[1,1,1,1,1,
        0,0,0,0,1,
        0,0,0,1,0,
        0,0,0,1,0,
        0,0,1,0,0,
        0,0,1,0,0,
        0,1,0,0,0,
        0,1,0,0,0,
        0,1,0,0,0]

eight =[0,1,1,1,0,
        1,0,0,0,1,
        1,0,0,0,1,
        1,0,0,0,1,
        0,1,1,1,0,
        1,0,0,0,1,
        1,0,0,0,1,
        1,0,0,0,1,
        0,1,1,1,0]

nine = [0,1,1,1,0,
        1,0,0,0,1,
        1,0,0,0,1,
        1,0,0,0,1,
        1,1,1,1,1,
        0,0,0,0,1,
        0,0,0,0,1,
        0,0,0,0,1,
        0,1,1,1,0]

# the proper formatted numbers
number_array = [zero, one, two, three, four, five, six, seven, eight, nine]

# creates a new file called Digits.txt
file = open('newDigits.txt', 'w')
complete_number_list = []

# seed random number generator
seed(9853)

# will make this many of each mutation type
set_count = 1000
mutation_const = 95
for number in number_array:
    new_number_list = []
    for _ in range(set_count):
        new_number = []
        for x in number:

            # if the number is 0
            if x == 0:

                # if the previous or next number is 1, roll mutation
                next_num = number[x+1]
                prev_num = number[x-1]
                if next_num == 1 or prev_num == 1:

                    # roll a percent from 0 to 100
                    rolled_val = randint(0, 100)

                    # mutation chance is 1 - mutation_const
                    if rolled_val > mutation_const:
                        new_number.append(1)
                    else:
                        new_number.append(0)
            else:
                new_number.append(1)
            new_number_list.append(new_number)
        complete_number_list.append(new_number)

file.write("--------- Complete List ---------" + '\n')
file.write(str(complete_number_list) + '\n')

# close the file
file.close()

