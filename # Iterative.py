# Iterative
n = 10
a, b = 0, 1
for _ in range(n):
    print(a, end=" ")
    a, b = b, a + b

# Recursive
def fib(n):
    return n if n <= 1 else fib(n-1) + fib(n-2)

def fact(n):
    return 1 if n <= 1 else n * fact(n-1)

# Iterative
res = 1
for i in range(1, n+1):
    res *= i