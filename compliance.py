def check_compliance(
    id_card,
    shirt_tucked,
    formal_trousers,
    formal_shoes,
    hair_ok,
    beard_ok
):
    rules = {
        "ID Card": id_card,
        "Shirt Tucked": shirt_tucked,
        "Formal Trousers": formal_trousers,
        "Formal Shoes": formal_shoes,
        "Hair": hair_ok,
        "Beard": beard_ok
    }

    issues = []

    for rule, status in rules.items():
        if not status:
            issues.append(rule)

    if len(issues) == 0:
        result = "COMPLIANT"
    else:
        result = "NOT COMPLIANT"

    return result, rules, issues