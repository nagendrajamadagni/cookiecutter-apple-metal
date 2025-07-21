import subprocess

subprocess.call(['git', 'init'])
subprocess.call(['git', 'add', 'README.md', 'LICENSE', '.gitignore'])
subprocess.call(['git', 'commit', '-m', 'Initial Commit'])
