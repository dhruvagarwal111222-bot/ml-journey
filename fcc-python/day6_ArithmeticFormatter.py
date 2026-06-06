def arithmetic_arranger(problems, show_answer=False):
    # Error: too many problems
    if len(problems) > 5:
        return "Error: Too many problems."

    top_row = []
    bottom_row = []
    dash_row = []
    answer_row = []

    for problem in problems:
        parts = problem.split()
        num1, operator, num2 = parts[0], parts[1], parts[2]

        # Error: invalid operator
        if operator not in ['+', '-']:
            return "Error: Operator must be '+' or '-'."

        # Error: numbers must only contain digits
        if not num1.isdigit() or not num2.isdigit():
            return "Error: Numbers must only contain digits."

        # Error: max four digits
        if len(num1) > 4 or len(num2) > 4:
            return "Error: Numbers cannot be more than four digits."

        # Calculate answer
        if operator == '+':
            answer = str(int(num1) + int(num2))
        else:
            answer = str(int(num1) - int(num2))

        # Width = longest number + 2 (operator + space)
        width = max(len(num1), len(num2)) + 2

        top_row.append(num1.rjust(width))
        bottom_row.append(operator + ' ' + num2.rjust(width - 2))
        dash_row.append('-' * width)
        answer_row.append(answer.rjust(width))

    arranged = '    '.join(top_row) + '\n' + '    '.join(bottom_row) + '\n' + '    '.join(dash_row)

    if show_answer:
        arranged += '\n' + '    '.join(answer_row)

    return arranged