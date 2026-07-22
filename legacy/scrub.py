import os
import re

directory = r"c:\DEV\Projects\College_Projects\motor-health-pso-cuda\docs\architecture"

replacements = {
    r"\*\*Role:\*\*(.*)": "**Authors:** Ravva Nagarjun",
    r"\*\*Document Owner:\*\*(.*)": "**Authors:** Ravva Nagarjun",
    r"\*\*Reviewers:\*\*(.*)": "",
    r"\*\*Review Board:\*\*(.*)": "",
    r"\*\*Board Members:\*\*(\n\*(.*))*": "",
    r"As your CTO, my job is to ensure we do not build": "The engineering team's goal is to ensure AeroForge does not become",
    r"Here is my evaluation of your 5 proposed innovations:": "Here is the architectural evaluation of the 5 proposed innovations:",
    r"As your CTO, I have evaluated": "The engineering team has evaluated",
    r"\* \*Verdict:\*": "* *Rationale:*",
    r"Your current implementation is": "The initial implementation is",
    r"we need to completely rethink": "the architecture completely rethinks",
    r"\*\*From:\*\* Principal AI Architect(.*)": "**Authors:** Ravva Nagarjun",
    r"\*\*To:\*\* R&D Engineering Team": "",
    r"\*\*Subject:\*\*": "**Topic:**",
}

for filename in os.listdir(directory):
    if filename.endswith(".md"):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        for pattern, replacement in replacements.items():
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

        # Additional hardcoded cleanups for specific strings
        content = content.replace("As your CTO, my job is to", "The goal is to")
        content = content.replace("my job is to", "the goal is to")
        content = content.replace("I have evaluated", "the team has evaluated")
        content = content.replace("Your task is to", "The task was to")
        content = content.replace("your 5 proposed innovations", "the 5 proposed innovations")
        
        # Strip multiple empty lines left behind by removing reviewers
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            
print("Documents successfully scrubbed.")
