from rps_playground.algorithms import ALL_ALGORITHM_CLASSES
import inspect
import sys
import os

# Add parent dir to path if needed (though running as module handles it usually)
sys.path.append('/Users/vaibhavsingh/fun')

output = []
for i, cls in enumerate(ALL_ALGORITHM_CLASSES):
    doc = inspect.getdoc(cls) or "No docstring"
    # Clean docstring indentation
    lines = [line.strip() for line in doc.splitlines()]
    clean_doc = "\n".join(lines)
    
    output.append(f"--- {i+1}. {cls.name} ({cls.__name__}) ---")
    output.append(clean_doc)
    output.append("")

with open("algo_audit.txt", "w") as f:
    f.write("\n".join(output))

print(f"Dumped {len(ALL_ALGORITHM_CLASSES)} algorithms to algo_audit.txt")
