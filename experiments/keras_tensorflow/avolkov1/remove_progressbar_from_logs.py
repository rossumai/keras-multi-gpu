import sys
from sacred.utils import apply_backspaces_and_linefeeds

with open(sys.argv[1], 'r') as input, open(sys.argv[2], 'w') as output:
   output.write(apply_backspaces_and_linefeeds(input.read()))
