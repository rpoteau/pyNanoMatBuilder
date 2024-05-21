import os

def create_html_from_structure(path):
    """
    Generate an HTML file to view the structure using JSmol.

    Args:
        path (str): The path to the structure file (XYZ or PDB).
    """
    # Extract the file name from the path
    file_name = os.path.basename(path)
    # Get the absolute path of the directory containing the structure file
    directory = os.path.dirname(path)

    # Get the absolute path of the application directory (where JSmol scripts are located)
    app_directory = os.path.dirname(os.path.abspath(__file__))
    jsmol_dir = os.path.join(app_directory, "jmol2", "jsmol")

    # Replace backslashes with double backslashes in file path for HTML content
    file_name_html = file_name.replace("\\", "\\\\")

    # Generate the HTML content, replacing backslashes with forward slashes for compatibility
    html_content = """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <title>xyz Viewer</title>
        <script src="{}"></script>
        <script src="{}"></script>
        <script>jmolInitialize("{}");</script>
    </head>
    <body>

    <script>
    jmolApplet(400, "load {} ", "0") 
    </script>
    </body>
    </html>
    """.format(
        os.path.join(jsmol_dir, 'JSmol.min.js').replace('\\', '/'),
        os.path.join(jsmol_dir, 'js', 'Jmol2.js').replace('\\', '/'),
        jsmol_dir.replace('\\', '/'),
        file_name_html
    )

    # Save the HTML content to a file named index.html in the same directory as the structure file
    html_file_path = os.path.join(directory, "index.html")
    print(html_file_path)
    with open(html_file_path, "w", encoding="utf-8") as html_file:
        html_file.write(html_content)