import subprocess

def invoke(filepath):
    result = subprocess.run(['python', filepath], capture_output=True, text=True)
    output = result.stdout.strip()
    return output