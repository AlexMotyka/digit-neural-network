from random import seed
from random import randint

# seed random number generator
seed(1337)

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

# will make this many of each mutation type
set_count = 1000

#### MUTATION SET 1 ####
# file.write("--------- Mutation 1's ---------" + '\n')
# count1 = 0
# for number in number_array:
#     new_number_list = []
#     for _ in range(set_count):
#         new_number = []
#         for x in number:
#
#             # if the number is a 0, then roll 0 to 100 to see if it mutates (> 90 means mutation)
#             if x == 0:
#                 rolled_val = randint(0, 100)
#                 if rolled_val > 95:
#                     new_number.append(1)
#                 else:
#                     new_number.append(0)
#             else:
#                 new_number.append(1)
#         new_number_list.append(new_number)
#         complete_number_list.append(new_number)
#     file.write("number" + str(count1) + '\n')
#     file.write(str(new_number_list) + '\n')
#     count1 = count1 + 1

#### MUTATION SET 2 ####
# file.write("--------- Mutation 0's ---------" + '\n')
# count2 = 0
# for number in number_array:
#     new_number_list = []
#     for _ in range(set_count):
#         new_number = []
#         for x in number:
#
#             # if the number is a 1, then roll 0 to 100 to see if it mutates (> 90 means mutation)
#             if x == 1:
#                 rolled_val = randint(0, 100)
#                 if rolled_val > 95:
#                     new_number.append(0)
#                 else:
#                     new_number.append(1)
#             else:
#                 new_number.append(0)
#         new_number_list.append(new_number)
#         complete_number_list.append(new_number)
#     file.write("number" + str(count2) + '\n')
#     file.write(str(new_number_list) + '\n')
#     count2 = count2 + 1

#### MUTATION SET 3 ####
file.write("--------- Mutation 0's and 1's ---------" + '\n')
count3 = 0
for number in number_array:
    new_number_list = []
    for _ in range(set_count):
        new_number = []
        for x in number:

            # Mutation chance on both 0 and 1
            if x == 1:
                rolled_val = randint(0, 100)
                if rolled_val > 95:
                    new_number.append(0)
                else:
                    new_number.append(1)
            elif x == 0:
                rolled_val = randint(0, 100)
                if rolled_val > 95:
                    new_number.append(1)
                else:
                    new_number.append(0)
        new_number_list.append(new_number)
        complete_number_list.append(new_number)
    file.write("number" + str(count3) + '\n')
    file.write(str(new_number_list) + '\n')
    count3 = count3 + 1

file.write("--------- Complete List ---------" + '\n')
file.write(str(complete_number_list) + '\n')

# close the file
file.close()

test = [0, 1, 1, 1, 0,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 1, 0, 0, 1,
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        0, 0, 0, 0, 1,
        0, 1, 1, 1, 0]
