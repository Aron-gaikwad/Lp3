import random
import time


# QuickSort with deterministic pivot (last element as pivot)
def quick_sort_deterministic(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[-1]  # Last element as the pivot
    left = [x for x in arr[:-1] if x <= pivot]
    right = [x for x in arr[:-1] if x > pivot]
    return quick_sort_deterministic(left) + [pivot] + quick_sort_deterministic(right)

# QuickSort with randomized pivot
def quick_sort_randomized(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[random.randint(0, len(arr) - 1)]  # Random element as the pivot
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort_randomized(left) + middle + quick_sort_randomized(right)

# Function to perform the sorting and compare times
def analyze_sorting():
    # Input: List of integers
    n = int(input("Enter number of elements: "))
    arr = [random.randint(1, 1000) for _ in range(n)]  # Generate random numbers
    print("Original array:", arr)
    
    # Deterministic QuickSort
    start = time.time()
    sorted_arr_deterministic = quick_sort_deterministic(arr.copy())
    end = time.time()
    time_deterministic = end - start
    print("\nDeterministic QuickSort result:", sorted_arr_deterministic)
    print(f"Time taken by Deterministic QuickSort: {time_deterministic:.6f} seconds")
    
    # Randomized QuickSort
    start = time.time()
    sorted_arr_randomized = quick_sort_randomized(arr.copy())
    end = time.time()
    time_randomized = end - start
    print("\nRandomized QuickSort result:", sorted_arr_randomized)
    print(f"Time taken by Randomized QuickSort: {time_randomized:.6f} seconds")

# Run the analysis
analyze_sorting()
