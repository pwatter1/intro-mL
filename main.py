
def reverse_string(str):
    left_index = 0
    right_index = len(str)-1

    my_list = list(str)

    while left_index < right_index:
        temp = my_list[left_index]
        my_list[left_index] = my_list[right_index]
        my_list[right_index] = temp

        left_index += 1
        right_index -= 1

    print("".join(my_list))

reverse_string("apples")


class Stack(object):
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        self.items.pop()

    def size(self):
        return len(self.items)

    def print_stack(self):
        for i in reversed(self.items):
            print(i)

    def get_top(self):
        print "Top of our Stack: ", self.items[-1]

s = Stack()
print s.isEmpty()
print ''
s.push(4)
s.push("dog")
s.push("APPLE")
s.push("Spotify")
s.print_stack()
s.get_top()


def check_balanced_parantheses(str):
    # '[ ( ) ( ) } '
    stack = []
    openings = "({["
    closings = ")}]"
    if len(str) < 2: return False

    for char in str:
        if char in openings:
            stack.append(char)

        elif char in closings:
            if(len(stack) == 0):
                return False
            else:
                pop = stack.pop()
                balancing_bracket = openings[closings.index(char)]
                if pop != balancing_bracket: return False
        else:
            return False

    return len(stack) == 0

print ""
print "Is string balanced? ", check_balanced_parantheses('(())')


iparens = iter('(){}[]<>')
parens = dict(zip(iparens, iparens))
closing = parens.values()

def balanced(astr):
    stack = []
    for c in astr:
        d = parens.get(c, None)
        if d:
            stack.append(d)
        elif c in closing:
            if not stack or c != stack.pop():
                return False
    return not stack

print ""
print balanced('(([[{}')

def factorial(num):
    ''' 
    if(num == 0): 
        return 1
    else: 
        return num * factorial(num -1)
    '''
    result = 1
    for index in xrange(1, num+1):
        result *= index
    return result


def fib(num):
    # recursive gets slow as num large
    if(n <= 1):
        return n
    else:
        return fib(n-1)+fib(n-2)
    # or map certain answers to hash map


def is_string_unique(str):
    if len(str) > 128: return False

    arr = [False] * 128

    for char in str:
        if arr[ord(char)] == False: 
            arr[ord(char)] = True
        else:
            return False

    return True

print is_string_unique("this")


def is_permutation(str1, str2):

    if(len(str1) != len(str2)): return False
    else: return ''.join(sorted(str1)) == ''.join(sorted(str2))



def urlify(str):
    lst = str.split()

    for char in range(len(lst)):
        lst[char] = lst[char]+'%20'

    print ''.join(lst)

urlify('this is an example baby spotify')


# is a string a permutation of a palindrome
# "aa bb cc f"
# "abcfcba"

import string
def is_permutation_of_pal(str):
    d = dict.fromkeys(string.ascii_lowercase, False)
    count = 0
    for char in str:
        if(ord(char) > 96 and ord(char) < 123):
            d[char] = not d[char]

    for key in d:
        if d[key] is True:
            count += 1
            if count > 1:
                return False
    
    return True


def string_compression(mystr):

    lst = sorted(mystr)
    d = dict.fromkeys(set(lst), 0)
    result = ''

    for char in lst:
        d[char] += 1

    for key, value in d.iteritems():
        result += (key + str(value))

    if len(result) > mystr:
        print mystr
    else: 
        print result

string_compression("aaaadddeppppppppie")

def compress(mystr):
    result = []
    sort = sorted(mystr)
    current = sort[0]
    count = 0

    for char in sort:
        if char == current:
            count += 1
        else:
            result.append(current + str(count))
            current = char
            count = 1

    # never reaches else after last
    result.append(current + str(count))  
    ret = ''.join(result)

    if(len(ret) > len(mystr)):
        print mystr
    else: 
        print ret

compress("aaaadddppppppppeie")

def print_permutations(lst, num, str):
	if(len(str) == num):
		print str
	else:
		for char in lst:
			print_permutations(lst, num, str+char)


print_permutations(['a','b','c'], 3, '')


def remove_duplicates(lst):
    seen = set()
    result = []

    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item+' ')

    return ''.join(result)

print remove_duplicates(['apple', 'apple', 'yeeet', 'cow'])


def find_missing_num(lst):
    # method - sort, loop through until index+1 doesnt equal next
    #           O(n log n) but space is O(1)
    # method - create boolean array and traverse NOT
    #           O(n) and space O(n) - decent
    # method above could be done with taking sums both

    totalXor = 0
    arrXor = 0

    for index in range(len(lst)+1, 1):
        totalXor ^= index
    for index in lst:
        arrXor ^= index

    return totalXor ^ arrXor

    # this is O(n) and O(1)

print find_missing_num([1,2,3,4,6,7])


def find_two_missing(lst):
    #[1,2,4] --> 3, 5
