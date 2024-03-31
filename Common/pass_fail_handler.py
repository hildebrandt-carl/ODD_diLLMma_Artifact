def read_pass_fail_file(filename):
    # Initialize empty lists for passing and failing IDs
    passing_ids = []
    failing_ids = []

    # Use a variable to track the current section
    current_section = None

    # Open the file for reading
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if line == "Passing FrameIDs":
                current_section = 'passing'
            elif line == "Failing FrameIDs":
                current_section = 'failing'
            elif line in ("================", ""):
                continue  # Ignore separators and empty lines
            else:
                # Add the ID to the correct list based on the current section
                if current_section == 'passing':
                    passing_ids.append(int(line))
                elif current_section == 'failing':
                    failing_ids.append(int(line))

    return passing_ids, failing_ids