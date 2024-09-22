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
        """Recursively compute the Time-to-Violation (TV) based on the given values."""
        if self.kind == 'prop':
            # Look up the TV value for the atomic proposition
            return tv_values[self.prop]
        elif self.kind == 'neg':
            # For negation, we now explicitly expect the negated form in the tv_values
            neg_prop = f'~{self.left.prop}'  # Representing negation as ~prop in tv_values
            if neg_prop not in tv_values:
                raise ValueError(f"TV for negation of {self.left.prop} (as ~{self.left.prop}) not provided in tv_values")
            return tv_values[neg_prop]
        elif self.kind == 'and':
            tv_left = self.left.compute_tv(tv_values)
            tv_right = self.right.compute_tv(tv_values)
            return min(tv_left, tv_right)
        elif self.kind == 'or':
            tv_left = self.left.compute_tv(tv_values)
            tv_right = self.right.compute_tv(tv_values)
            return max(tv_left, tv_right)


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
            right = parse_nnf_formula(input_str[i+1:])
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
# Including the negated versions (~a, ~b, ..., ~o)
inf = float('inf')
tv_values = {
    'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0, 'f': 0, 'g': 0, 'h': 0, 'i': 0,
    'j': 0, 'k': 0, 'l': 0, 'm': 0, 'n': 0, 'o': inf,
    '~a': 3, '~b': 3, '~c': 3, '~d': 3, '~e': 3, '~f': 3, '~g': inf,
    '~h': inf, '~i': inf, '~j': 7, '~k': inf, '~l': 9, '~m': 9, '~n': 7, '~o': 0
}

# Compute TV for the formula in NNF
import time
iniT = time.time()
for _ in range(1000):
    tv_result = parsed_nnf_formula.compute_tv(tv_values)
print((time.time() - iniT) / 1000)
print(f"The Time-to-Violation (TV) for the formula is: {tv_result}")
