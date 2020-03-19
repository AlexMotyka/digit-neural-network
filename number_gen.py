from random import seed
from random import randint
import os

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
file = open('correct_digits.txt', 'w')
complete_number_list = []

# seed random number generator
seed(9853)

# will make this many of each mutation type
set_count = 50
mutation_const = 75
count = 0
os.mkdir('digits_correct')
for number in number_array:
    ind_file = open('digits_correct/' + str(count) + '_digits_correct.txt', 'w' )
    new_number_list = []
    for _ in range(set_count):
        new_number_removed = []
        new_number_added = []
        removed_mutated_flag = 0
        added_mutated_flag = 0
        for index, value in enumerate(number):

            # if the number is 0
            if value == 1:

                # can only change 1 --> 0 if its an 'end' digit
                left_num = False
                if index > 0:
                    left_num = bool(number[index - 1] == 1)
                up_num = False
                if index > 4:
                    up_num = bool(number[index - 5] == 1)
                down_num = False
                if index < 39:
                    down_num = bool(number[index + 5] == 1)

                right_num = False
                if index < 44:
                    right_num = bool(number[index + 1] == 1)

                if ((left_num and not right_num and not up_num and not down_num) or
                    (right_num and not left_num and not up_num and not down_num) or
                    (down_num and not up_num and not left_num and not right_num) or
                    (up_num and not down_num and not right_num and not left_num)) and removed_mutated_flag == 0:

                    # roll a percent from 0 to 100
                    rolled_val = randint(0, 100)

                    # mutation chance is 1 - mutation_const
                    if rolled_val > mutation_const:
                        new_number_removed.append(0)
                        removed_mutated_flag = 1
                    else:
                        new_number_removed.append(1)
                else:
                    new_number_removed.append(1)
            else:
                new_number_removed.append(0)
        for index, value in enumerate(number):

            # if the number is 0
            if value == 0:

                # can only change 1 --> 0 if its an 'end' digit
                left_num = False
                if index > 0:
                    left_num = bool(number[index - 1] == 1)
                up_num = False
                if index > 4:
                    up_num = bool(number[index - 5] == 1)
                down_num = False
                if index < 39:
                    down_num = bool(number[index + 5] == 1)

                right_num = False
                if index < 44:
                    right_num = bool(number[index + 1] == 1)

                if (left_num or right_num or up_num or down_num) and added_mutated_flag == 0:

                    # roll a percent from 0 to 100
                    rolled_val = randint(0, 100)

                    # mutation chance is 1 - mutation_const
                    if rolled_val > mutation_const:
                        new_number_added.append(1)
                        added_mutated_flag = 1
                    else:
                        new_number_added.append(0)
                else:
                    new_number_added.append(0)
            else:
                new_number_added.append(1)
        new_number_list.append(new_number_removed)
        new_number_list.append(new_number_added)
        complete_number_list.append(new_number_removed)
        complete_number_list.append(new_number_added)

    count = count + 1
    ind_file.write(str(new_number_list))
    ind_file.close()

file.write("--------- Complete List ---------" + '\n')
file.write(str(complete_number_list) + '\n')

# close the file
file.close()

test = [0, 1, 1, 1, 0,
        1, 0, 0, 0, 1,
        0, 0, 0, 0, 1,
        0, 0, 0, 0, 1,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1,
        0, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        0, 0, 1, 1, 0]