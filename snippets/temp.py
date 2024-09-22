import time
import sys

class NumberContainer:
    def __init__(self, numbers):
        self.numbers = numbers

def create_list_of_lists(n, m):
    return [list(range(m))] * n

def create_list_of_objects(n, m):
    return [NumberContainer(list(range(m))) for _ in range(n)]

def read_list_of_lists(data):
    total = 0
    for inner_list in data:
        for num in inner_list:
            total += num
    return total

def read_list_of_objects(data):
    total = 0
    for obj in data:
        for num in obj.numbers:
            total += num
    return total

def benchmark(n, m):
    # List of Lists
    start_time = time.time()
    list_of_lists = create_list_of_lists(n, m)
    create_time_lists = time.time() - start_time

    start_time = time.time()
    read_list_of_lists(list_of_lists)
    read_time_lists = time.time() - start_time

    # List of Objects
    start_time = time.time()
    list_of_objects = create_list_of_objects(n, m)
    create_time_objects = time.time() - start_time

    start_time = time.time()
    read_list_of_objects(list_of_objects)
    read_time_objects = time.time() - start_time

    print(f"List of Lists - Create: {create_time_lists:.6f}s, Read: {read_time_lists:.6f}s")
    print(f"List of Objects - Create: {create_time_objects:.6f}s, Read: {read_time_objects:.6f}s")
    print(f"Memory usage (List of Lists): {sys.getsizeof(list_of_lists)} bytes")
    print(f"Memory usage (List of Objects): {sys.getsizeof(list_of_objects)} bytes")

if __name__ == "__main__":
    n = 100000  # number of lists/objects
    m = 14      # number of elements in each inner list
    benchmark(n, m)