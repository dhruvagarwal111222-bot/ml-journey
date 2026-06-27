def add_time(start, duration, day=None):

    # String splitting — Cipher module
    time_part, period = start.split()
    start_hour, start_min = [int(x) for x in time_part.split(':')]  # list comprehension module
    dur_hour, dur_min = [int(x) for x in duration.split(':')]

    # Convert to 24-hour — arithmetic from Luhn module
    if period == 'PM' and start_hour != 12:
        start_hour += 12
    elif period == 'AM' and start_hour == 12:
        start_hour = 0

    # Modulo + integer division — Luhn module
    total_min = start_hour * 60 + start_min + dur_hour * 60 + dur_min
    days_passed = total_min // (24 * 60)
    remaining = total_min % (24 * 60)

    new_hour = remaining // 60
    new_min = remaining % 60

    # Back to 12-hour format
    if new_hour == 0:
        result_hour, result_period = 12, 'AM'
    elif new_hour < 12:
        result_hour, result_period = new_hour, 'AM'
    elif new_hour == 12:
        result_hour, result_period = 12, 'PM'
    else:
        result_hour, result_period = new_hour - 12, 'PM'

    # f-string formatting — Arithmetic Formatter project
    result = f'{result_hour}:{new_min:02d} {result_period}'

    # List comprehension + modulo for day lookup — list comprehension module + Luhn
    if day:
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        start_index = [d.lower() for d in days].index(day.lower())
        new_day = days[(start_index + days_passed) % 7]
        result += f', {new_day}'

    # Days later — string manipulation module
    if days_passed == 1:
        result += ' (next day)'
    elif days_passed > 1:
        result += f' ({days_passed} days later)'

    return result