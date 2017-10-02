from sacred.utils import apply_backspaces_and_linefeeds

with open('original_7gforce_results.txt', 'r') as input,
  open('original_7gforce_results_nocr.txt', 'w') as output:
   output.write(apply_backspaces_and_linefeeds(input.read()))
