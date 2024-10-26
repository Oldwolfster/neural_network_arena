def format_weighted_term(weight, input_val):
    # Use scientific notation for large/small numbers; otherwise, fixed-point formatting
    if abs(weight) >= 1e4 or abs(weight) < 1e-4:
        formatted_weight = f"{weight:>8.2e}"  # Scientific notation with 2 decimal places, width 8
    else:
        formatted_weight = f"{weight:>7.3f}"  # Fixed-point notation with 3 decimal places, width 7

    if abs(input_val) >= 1e4 or abs(input_val) < 1e-4:
        formatted_input = f"{input_val:>8.2e}"
    else:
        formatted_input = f"{input_val:>7.3f}"

    # Calculate and format result with similar logic for scientific or fixed-point formatting
    result = weight * input_val
    if abs(result) >= 1e4 or abs(result) < 1e-4:
        formatted_result = f"{result:>8.2e}"
    else:
        formatted_result = f"{result:>7.3f}"

    # Combine them with aligned operators
    return f"{formatted_weight} * {formatted_input} = {formatted_result}"

# Example usage with various ranges
print(format_weighted_term(0.1, 3.083))         # Smaller values
print(format_weighted_term(1E8, 3.083))         # Very large weight
print(format_weighted_term(1E-8, 3.083))        # Very small weight
print(format_weighted_term(12345.6789, 0.0001)) # Mixed scale values
