from compliance import check_compliance

result, issues = check_compliance(
    id_card=False,
    shirt_tucked=False,
    formal_trousers=False,
    formal_shoes=True,
    hair_ok=True,
    beard_ok=True
)

print("Result:", result)

if issues:
    print("Issues:")
    for issue in issues:
        print("-", issue)