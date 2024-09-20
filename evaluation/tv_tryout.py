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

    def to_nnf(self):
        """Convert the formula to Negation Normal Form (NNF)."""
        if self.kind == 'prop':
            return self  # Atomic propositions are already in NNF
        elif self.kind == 'neg':
            # Apply negation rules
            if self.left.kind == 'prop':
                return self  # Negation of atomic proposition is in NNF
            elif self.left.kind == 'neg':
                # Double negation: ¬(¬φ) -> φ
                return self.left.left.to_nnf()
            elif self.left.kind == 'and':
                # ¬(φ1 ∧ φ2) -> ¬φ1 ∨ ¬φ2
                return Formula('or', Formula('neg', self.left.left).to_nnf(), Formula('neg', self.left.right).to_nnf())
            elif self.left.kind == 'or':
                # ¬(φ1 ∨ φ2) -> ¬φ1 ∧ ¬φ2
                return Formula('and', Formula('neg', self.left.left).to_nnf(), Formula('neg', self.left.right).to_nnf())
        elif self.kind == 'and':
            return Formula('and', self.left.to_nnf(), self.right.to_nnf())
        elif self.kind == 'or':
            return Formula('or', self.left.to_nnf(), self.right.to_nnf())

    def compute_tv(self, tv_values):
        """Recursively compute the Time-to-Violation (TV) based on the given values."""
        if self.kind == 'prop':
            return tv_values[self.prop]
        elif self.kind == 'neg':
            tv_left = self.left.compute_tv(tv_values)
            return float('inf') if tv_left == float('inf') else tv_left
        elif self.kind == 'and':
            tv_left = self.left.compute_tv(tv_values)
            tv_right = self.right.compute_tv(tv_values)
            return min(tv_left, tv_right)
        elif self.kind == 'or':
            tv_left = self.left.compute_tv(tv_values)
            tv_right = self.right.compute_tv(tv_values)
            return max(tv_left, tv_right)


def parse_formula(input_str):
    """Parse the formula string and return a structured Formula object."""
    # Remove spaces from the input string
    input_str = input_str.replace(' ', '')

    # Handle atomic propositions
    if re.match(r'^[a-z]$', input_str):
        return Formula('prop', prop=input_str)

    # Handle negations
    if input_str.startswith('¬'):
        return Formula('neg', left=parse_formula(input_str[1:]))

    # Find the main operator in the formula
    depth = 0
    for i, char in enumerate(input_str):
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        elif depth == 0 and char in {'∧', '∨', '→'}:
            left = parse_formula(input_str[:i])
            right = parse_formula(input_str[i + 1:])
            if char == '∧':
                return Formula('and', left=left, right=right)
            elif char == '∨':
                return Formula('or', left=left, right=right)
            elif char == '→':
                # Implication φ → ψ is equivalent to ¬φ ∨ ψ
                return Formula('or', left=Formula('neg', left=left), right=right)

    # Remove surrounding parentheses
    if input_str.startswith('(') and input_str.endswith(')'):
        return parse_formula(input_str[1:-1])

    raise ValueError(f"Invalid formula format: {input_str}")


# Example input string
input_formula_str = "((a ∨ b ∨ c ∨ d ∨ e ∨ f ∨ g ∨ h ∨ i) → ((j → (¬k ∧ ¬l)) ∧ (m → ¬n)) ∨ ¬o)"

# Parse the formula string into a structured Formula object
parsed_formula = parse_formula(input_formula_str)

# Example TV values for each atomic proposition
tv_values = {
    'a': 5,
    'b': 3,
    'c': 7,
    'd': 4,
    'e': 6,
    'f': 8,
    'g': 9,
    'h': 2,
    'i': 10,
    'j': 6,
    'k': 4,
    'l': 3,
    'm': 5,
    'n': 7,
    'o': 8
}

# Convert formula to NNF
nnf_formula = parsed_formula.to_nnf()

# Compute TV for the formula in NNF
import time
iniT = time.time()
for _ in range(1000):
    tv_result = nnf_formula.compute_tv(tv_values)
print((time.time() - iniT)/1000)
print(f"NNF Formula: {nnf_formula}")
print(f"The Time-to-Violation (TV) for the formula is: {tv_result}")
