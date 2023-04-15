import sys
import re
from pprint import pprint

if len(sys.argv) < 2:
    print("Please provide a file path.")
    sys.exit()

file_path = sys.argv[1]

with open(file_path, 'r') as file:
    input_lines = file.readlines()

input_str = "".join(input_lines)
pattern = r'func.name\s*=\s*"([^"]*)"'
result = re.findall(pattern, input_str)
replace_dict = {"arg" + str(i): result[i] for i in range(len(result))}
pprint(replace_dict)

pattern = re.compile(r'\b(' + '|'.join(replace_dict.keys()) + r')\b')

with open(file_path, 'w') as file:
    for line in input_lines:
        line = pattern.sub(lambda x: replace_dict[x.group()], line)
        file.write(line)
