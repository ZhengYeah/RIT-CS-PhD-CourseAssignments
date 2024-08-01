from math import ceil, sqrt

# Function to calculate modular exponentiation
def modexp(base, exponent, modulus):
    result = 1
    base = base % modulus
    while exponent > 0:
        if exponent % 2 == 1:
            result = (result * base) % modulus
        exponent = exponent >> 1
        base = (base * base) % modulus
    return result

# BSGS algorithm to find x such that a^x = b mod m
def bsgs(a, b, m):
    n = ceil(sqrt(m))
    
    # Precompute baby steps
    baby_steps = {}
    val = 1
    for j in range(n):
        if val not in baby_steps:
            baby_steps[val] = j
        val = (val * a) % m
    
    # Compute giant steps
    giant_step = modexp(a, n * (m - 2), m)
    giant_val = b
    for i in range(n):
        if giant_val in baby_steps:
            return i * n + baby_steps[giant_val]
        giant_val = (giant_val * giant_step) % m
    
    return None

if __name__ == "__main__":
    a = 68093
    b = 836856 
    m = 10000019 

    x = bsgs(a, b, m)

    if x is not None:
        print("x =", x)
    else:
        print("No solution found.")

    print(f"Validation: {modexp(a, x, m)} = {b}")
