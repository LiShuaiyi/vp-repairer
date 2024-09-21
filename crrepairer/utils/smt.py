import sympy as sp


def stl2sympy(input_formula: str):
    return (
        input_formula.replace("and", "&")
        .replace("or", "|")
        .replace("!", "~")
        .replace("implies", ">>")
    )

def construct_cnf(stl_formula):
    """
    Construct Conjunctive Normal Form (CNF) using sympy - first needs to convert the formula to sp's interface.
    """
    if isinstance(stl_formula, str):
        sp_formula = stl2sympy(stl_formula)
    else:
        sp_formula = stl_formula
    cnf_formula = str(sp.to_cnf(sp_formula))
    return cnf_formula

def construct_dnf(stl_formula):
    """
    Construct Disjunctive Normal Form (DNF) using sympy - first needs to convert the formula to sp's interface.
    """
    if isinstance(stl_formula, str):
        sp_formula = stl2sympy(stl_formula)
    else:
        sp_formula = stl_formula
    dnf_formula = str(sp.to_dnf(sp_formula))
    return dnf_formula

def construct_nnf(stl_formula):
    """
    Construct Negation Normal Form (NNF) using sympy - first needs to convert the formula to sp's interface.
    """
    sp_formula = stl2sympy(stl_formula)
    nnf_formula = sp.to_nnf(sp.simplify(sp_formula))
    return nnf_formula