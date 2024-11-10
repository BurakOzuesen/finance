
def apply_functions_consecutively(variable, functions):
    result = variable
    for func in functions:
        result = func(result)
    return result