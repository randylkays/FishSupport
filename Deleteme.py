import os

print("Hello world")
files_path = os.path.realpath(__file__)
print("files_path:",files_path)

abs_path = os.path.abspath(__file__)
print("abs_path:",abs_path)