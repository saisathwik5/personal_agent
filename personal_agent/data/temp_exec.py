import os

# Create PDF content strings for Problems 1 and 4
problem_1_content = "Solution to Problem 1: [Your solution here]."
problem_4_content = "Solution to Problem 4: [Your solution here]."

# Path for the PDF
pdf_path = 'Problems_1_and_4_Solutions.txt'

# Write the solutions to a text file as a substitute for PDF creation
with open(pdf_path, 'w') as f:
    f.write(problem_1_content + '\n' + problem_4_content)