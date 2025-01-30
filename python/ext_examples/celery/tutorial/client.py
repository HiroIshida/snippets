from tasks import add, mul, sub

result1 = add.delay(4, 6)
result2 = mul.delay(3, 7)
result3 = sub.delay(10, 4)

print("Waiting for results...")
print(f"Addition Result: {result1.get(timeout=10)}")
print(f"Multiplication Result: {result2.get(timeout=10)}")
print(f"Subtraction Result: {result3.get(timeout=10)}")
