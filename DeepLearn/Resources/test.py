import requests
from bs4 import BeautifulSoup

def decode_message(url):
    html = requests.get(url).text
    parsed = BeautifulSoup(html, "html.parser")
    points = []

    for row in parsed.select("table tr"):
        cells = row.find_all("td")

        #skip header
        if "x" in cells[0].get_text().lower() or "x" in cells[1].get_text().lower() or "x" in cells[2].get_text().lower():
            continue

        x = int(cells[0].get_text())
        y = int(cells[2].get_text())
        character_val = cells[1].get_text()

        points.append((character_val, x, y))

    if not points:
        print("No data found.")
        return

    max_x = max(x for _, x, _ in points)
    max_y = max(y for _, _, y in points)

    #empty text area
    grid = [[" " for _ in range(max_x + 1)] for _ in range(max_y + 1)]

    for char, x, y in points:
        grid[y][x] = char

    for row in grid:
        print("".join(row))

decode_message("https://docs.google.com/document/d/e/2PACX-1vSvM5gDlNvt7npYHhp_XfsJvuntUhq184By5xO_pA4b_gCWeXb6dM6ZxwN8rE6S4ghUsCj2VKR21oEP/pub")