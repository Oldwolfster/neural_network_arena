import requests
from bs4 import BeautifulSoup

def fetch_google_doc_as_tuples(doc_url):

    try:
        # Fetch the content of the Google Doc
        response = requests.get(doc_url)
        response.raise_for_status()  # ensure the request was successful

        content = response.text
        parsed_content = BeautifulSoup(content, "html.parser")  # library to parse html table
        table = parsed_content.find("table")

        if not table:
            raise ValueError("No table found")

        rows = table.find_all("tr")
        cypher = []
        for i, row in enumerate(rows):
            if i == 0:
                continue    # Skip first header row
            cells = row.find_all("td")
            if cells:  # Only include populated cells
                # add to list as a tuple of 3 values, x, char, and y
                cypher.append(tuple(cell.get_text(strip=True) for cell in cells))

        return cypher

    except requests.exceptions.RequestException as e:
        print(f"Error fetching Google Doc: {e}")
        return []


def check_coordinate(x, y, txt):
    for t in txt:
        if t[0] == str(x) and t[-1] == str(y):  # is it the right coordinate
            print(f"Matched: ({x}, {y}) -> {t[1]}")
            return t[1]  # then return the char
    return None  # If coordinate not found return none.


def decode_cypher_text(txt):                    # while perhaps not the mmost efficent a simple approach
    last_col = max(int(t[0]) for t in txt)      # get the highest x coordinate
    last_row = max(int(t[-1]) for t in txt)     # get the highest y coordinate

    clr_text = []
    for y in range(last_row, -1, -1):           # Loop through each line(row) but 0 starts at bottom
        single_row=''
        for x in range(last_col + 1):           # loop through each column, constructing a line.
            coord=check_coordinate(x,y,txt)     # admittedly we pass the pointer to this string a lot but beats a global
            if coord == None:
                single_row += ' '
            else:
                single_row += coord
        if single_row:                          # Only append non-empty rows
            #print(f'Single row {y} = {single_row}')
            clr_text.append(single_row)
    return clr_text


#doc_url = "https://docs.google.com/document/d/e/2PACX-1vRMx5YQlZNa3ra8dYYxmv-QIQ3YJe8tbI3kqcuC7lQiZm-CSEznKfN_HYNSpoXcZIV3Y_O3YoUB1ecq/pub"
doc_url = "https://docs.google.com/document/d/e/2PACX-1vQGUck9HIFCyezsrBSnmENk5ieJuYwpt7YHYEzeNJkIb9OSDdx-ov2nRNReKQyey-cwJOoEKUhLmN9z/pub"
cypher_text = fetch_google_doc_as_tuples(doc_url)   # returns list of 3 part tuples (x, char, y)
clear_text = decode_cypher_text(cypher_text)        # decodes text
print("\n".join(clear_text))                        # prints text
