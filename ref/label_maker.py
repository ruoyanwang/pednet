import os
import glob
import random

proposal_names = sorted(glob.glob('./proposals/*png'))

print proposal_names[0][2:]

text_file = open("proposal.txt", "w+")

for name in proposal_names:
    text_file.write(name[2:]+' 0\n')

text_file.close()

