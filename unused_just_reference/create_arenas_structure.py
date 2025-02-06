import os

def create_directory_structure(base_path):
    structure = [
        'binary_decision\\single_input',
        'binary_decision\\two_inputs',
        'binary_decision\\multi_input_linear',
        'binary_decision\\multi_input_nonlinear',
        'binary_decision\\no_noise',
        'binary_decision\\low_noise',
        'binary_decision\\high_noise',
        'regression\\single_input',
        'regression\\two_inputs',
        'regression\\multi_input_linear',
        'regression\\multi_input_nonlinear',
        'regression\\no_noise',
        'regression\\low_noise',
        'regression\\high_noise',
    ]

    # Create the directories
    for path in structure:
        full_path = os.path.join(base_path, path)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created directory: {full_path}")

# Run the function with the path to `src\\arenas`
create_directory_structure('src\\arenas')
