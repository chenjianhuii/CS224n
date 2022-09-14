# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.
from utils import evaluate_places

if __name__ == "__main__":
    length = 500
    pred = ['London'] * length
    total, correct = evaluate_places("birth_dev.tsv", pred)
    print(f'Accuracy of london baseline is {correct*100/total}%')

