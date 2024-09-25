def transform_column_2(value):
    return value.upper()

def transform_column_5(value):
    return f"**{value}**"

fields = ['Field1', 'Field2', 'Field3', 'Field4', 'Field5', 'Field6', 'Field7']

# Apply transformations to columns 2 and 5, and omit columns 3 (index 2) and 6 (index 5)
transformed_fields = [
    transform_column_2(field) if i == 1 else
    transform_column_5(field) if i == 4 else
    field
    for i, field in enumerate(fields) if i not in {2, 5}  # Omit columns 3 (index 2) and 6 (index 5)
]

print(transformed_fields)
# Output: ['Field1', 'FIELD2', 'Field4', '**Field5**', 'Field7']
