#!/usr/bin/env python
from subprocess import run, PIPE
from pathlib import Path


dirpath = "D:/PythonDataScience/shellDataProcessing/"
# Find all the python files you created and print them out
for i in range(3):
    pppt = f"{dirpath}filegg_{i}.py"
    path = Path(pppt)
    path.write_text("#!/usr/bin/env python\n")
    path.write_text("#!/usr/bin/env python\n\nimport datetime\nprint(datetime.datetime.now())\nprint('Wonderful')")
   
  

# Find all the python files you created and print them out
for file in Path(r"D:/PythonDataScience/shellDataProcessing/").glob("*file*.py"):
    # gets the resolved full path
    fullpath = str(file.resolve())
    if "44" in fullpath:
        proc = run(["python", fullpath, "Mandela"], stdout=PIPE)
    else:
        proc = run(["python", fullpath], stdout=PIPE)
    print(proc)
    
    
