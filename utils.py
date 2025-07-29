from numpy import max

def normalise(numerical_list, floor, ceiling):

# A Simple Min-Max Feature Scaling Algorithm

    new_numerical_list = numerical_list.copy()
    minimum = min(new_numerical_list)
    maximum = max(new_numerical_list)
    total = sum(new_numerical_list)
    if total != 0 and maximum != minimum:
        for x in range(len(new_numerical_list)):
            new_numerical_list[x] = floor + (ceiling-floor)  * (new_numerical_list[x] - minimum) / (maximum - minimum)
        return new_numerical_list     
    else:
        return [1/len(new_numerical_list)] * len(new_numerical_list)


def cum_probs(numerical_list):

# A Simple Cumulative Distribution Function

    new_numerical_list = numerical_list.copy()
    total = sum(new_numerical_list)
    for x in range(len(new_numerical_list)):
        new_numerical_list[x] /= total
    for y in range(1, len(new_numerical_list)):
        new_numerical_list[y] += new_numerical_list[y-1]
    return new_numerical_list

def q_update(learning_rate, discount_factor, reward, q_table, current_val):
    maximum = max(q_table)
    return current_val + learning_rate * (reward + discount_factor * maximum - current_val)
    