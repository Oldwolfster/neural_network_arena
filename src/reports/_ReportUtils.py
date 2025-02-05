def clean_multiline_string(input_string):
    # Split the input string into lines
    lines = input_string.splitlines()

    # Strip leading and trailing spaces from each line
    cleaned_lines = [line.strip() for line in lines]

    # Join the cleaned lines back together with newlines
    cleaned_string = '\n'.join(cleaned_lines)

    return cleaned_string
"""
# Example usage
input_string = "" "
If all activations look the same for every input, something is wrong.
                  If hidden layer activations are all close to 1 or -1, sigmoid/tanh is saturating.
                    If output is always ~0.5, we have bad weight initialization. 
"" "

cleaned_string = clean_multiline_string(input_string)
print(cleaned_string)
"""