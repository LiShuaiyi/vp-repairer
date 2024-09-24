import re


class Formula:
    def __init__(self, kind, left=None, right=None, prop=None):
        self.kind = kind  # Can be 'prop', 'neg', 'and', 'or'
        self.left = left  # Left operand for binary operators
        self.right = right  # Right operand for binary operators
        self.prop = prop  # Atomic proposition (for 'prop' type)

    def __repr__(self):
        if self.kind == 'prop':
            return self.prop
        elif self.kind == 'neg':
            return f'¬{self.left}'
        elif self.kind == 'and':
            return f'({self.left} ∧ {self.right})'
        elif self.kind == 'or':
            return f'({self.left} ∨ {self.right})'

    def compute_tv(self, tv_values: dict):
        """Recursively compute the Time-to-Violation (TV) series based on the given time series."""

        # Handle atomic propositions
        if self.kind == 'prop':
            # Get the time series for the proposition
            return tv_values[self.prop]

        # Handle negation
        elif self.kind == 'neg':
            neg_prop = f'~{self.left.prop}'  # Representing negation as ~prop in tv_values
            if neg_prop not in tv_values:
                raise ValueError(
                    f"TV for negation of {self.left.prop} (as ~{self.left.prop}) not provided in tv_values")
            return tv_values[neg_prop]

        # Handle conjunction (element-wise min between left and right time series)
        elif self.kind == 'and':
            tv_left = self.left.compute_tv(tv_values)
            tv_right = self.right.compute_tv(tv_values)
            return [min(l, r) for l, r in zip(tv_left, tv_right)]

        # Handle disjunction (element-wise max between left and right time series)
        elif self.kind == 'or':
            tv_left = self.left.compute_tv(tv_values)
            tv_right = self.right.compute_tv(tv_values)
            return [max(l, r) for l, r in zip(tv_left, tv_right)]


def parse_nnf_formula(input_str):
    """Parse the NNF formula string and return a structured Formula object."""
    # Remove spaces from the input string
    input_str = input_str.replace(' ', '')

    # Handle atomic propositions
    if re.match(r'^~?[a-z]$', input_str):
        if input_str.startswith('~'):
            return Formula('neg', left=Formula('prop', prop=input_str[1:]))
        else:
            return Formula('prop', prop=input_str)

    # Find the main operator in the formula
    depth = 0
    for i, char in enumerate(input_str):
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        elif depth == 0 and char in {'&', '|'}:
            left = parse_nnf_formula(input_str[:i])
            right = parse_nnf_formula(input_str[i + 1:])
            if char == '&':
                return Formula('and', left=left, right=right)
            elif char == '|':
                return Formula('or', left=left, right=right)

    # Remove surrounding parentheses
    if input_str.startswith('(') and input_str.endswith(')'):
        return parse_nnf_formula(input_str[1:-1])

    raise ValueError(f"Invalid formula format: {input_str}")


# Example input string for the NNF format
input_nnf_formula_str = "~o | (~j & ~m) | (~j & ~n) | (~k & ~l & ~m) | (~k & ~l & ~n) | (~a & ~b & ~c & ~d & ~e & ~f & ~g & ~h & ~i)"

# Parse the NNF formula string into a structured Formula object
parsed_nnf_formula = parse_nnf_formula(input_nnf_formula_str)

# Example TV values for each atomic proposition
# Now using time series instead of single values
inf = float('inf')
tv_values = {'a': [0, 1, 2, inf, inf, inf, 6, 7, 8, 9, 10, 11, 12, 13, 14],
             'b': [0, 1, 2, inf, inf, inf, 6, 7, 8, 9, 10, 11, 12, 13, 14],
             'c': [0, 1, 2, inf, inf, inf, 6, 7, 8, 9, 10, 11, 12, 13, 14],
             'd': [0, 1, 2, inf, inf, inf, inf, inf, inf, inf, inf, 11, 12, 13, 14],
             'e': [0, 1, 2, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf],
             'f': [0, 1, 2, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf],
             'g': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
             'h': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
             'i': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
             'j': [0, 1, 2, 3, 4, 5, 6, inf, inf, inf, inf, inf, inf, inf, inf],
             'k': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
             'l': [0, 1, 2, 3, 4, 5, 6, 7, 8, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 19],
             'm': [0, 1, 2, 3, 4, 5, 6, 7, 8, inf, inf, inf, inf, inf, 14],
             'n': [0, 1, 2, 3, 4, 5, 6, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf],
             'o': [inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf],
             '~a': [inf, inf, inf, 3, 4, 5, inf, inf, inf, inf, inf, inf, inf, inf, inf],
             '~b': [inf, inf, inf, 3, 4, 5, inf, inf, inf, inf, inf, inf, inf, inf, inf],
             '~c': [inf, inf, inf, 3, 4, 5, inf, inf, inf, inf, inf, inf, inf, inf, inf],
             '~d': [inf, inf, inf, 3, 4, 5, 6, 7, 8, 9, 10, inf, inf, inf, inf],
             '~e': [inf, inf, inf, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
             '~f': [inf, inf, inf, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
             '~g': [inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf],
             '~h': [inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf],
             '~i': [inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf],
             '~j': [inf, inf, inf, inf, inf, inf, inf, 7, 8, 9, 10, 11, 12, 13, 14],
             '~k': [inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf],
             '~l': [inf, inf, inf, inf, inf, inf, inf, inf, inf, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, inf],
             '~m': [inf, inf, inf, inf, inf, inf, inf, inf, inf, 9, 10, 11, 12, 13, inf],
             '~n': [inf, inf, inf, inf, inf, inf, inf, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
             '~o': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]}
# tv_values = {
#     'a': [0, 1, 0, 0, 0],
#     'b': [0, 0, 0, 1, 0],
#     'c': [0, 0, 0, 0, 1],
#     'd': [0, 0, 1, 0, 0],
#     'e': [0, 0, 0, 0, 0],
#     'f': [0, 0, 0, 0, 0],
#     'g': [0, 0, 0, 0, 0],
#     'h': [0, 0, 0, 0, 0],
#     'i': [0, 0, 0, 0, 0],
#     'j': [0, 0, 0, 0, 0],
#     'k': [0, 0, 0, 0, 0],
#     'l': [0, 0, 0, 0, 0],
#     'm': [0, 0, 0, 0, 0],
#     'n': [0, 0, 0, 0, 0],
#     'o': [inf, inf, inf, inf, inf],
#     '~a': [3, 0, 0, 0, 0],
#     '~b': [3, 0, 0, 0, 0],
#     '~c': [3, 0, 0, 0, 0],
#     '~d': [3, 0, 0, 0, 0],
#     '~e': [3, 0, 0, 0, 0],
#     '~f': [3, 0, 0, 0, 0],
#     '~g': [inf, inf, inf, inf, inf],
#     '~h': [inf, inf, inf, inf, inf],
#     '~i': [inf, inf, inf, inf, inf],
#     '~j': [7, 0, 0, 0, 0],
#     '~k': [inf, inf, inf, inf, inf],
#     '~l': [9, 0, 0, 0, 0],
#     '~m': [9, 0, 0, 0, 0],
#     '~n': [7, 0, 0, 0, 0],
#     '~o': [0, 0, 0, 0, 0]
# }

# Compute TV for the formula in NNF
import time

iniT = time.time()
for _ in range(1000):
    tv_result = parsed_nnf_formula.compute_tv(tv_values)
print((time.time() - iniT) / 1000)
print(f"The Time-to-Violation (TV) for the formula is: {tv_result}")
