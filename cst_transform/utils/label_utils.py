
def _is_correct(result):
    check = "true" if result['ground_truth'] else "false"
    return check in result["status"]


def result_to_correct_labels(result_dict, tool_order=None):

    # We are only looking at reachability
    if 'reachability' not in result_dict:
        raise ValueError("Expected a dict with reachability info but got %s" % str(result_dict.keys()))
    result_dict = result_dict["reachability"]

    if tool_order is None:
        tool_order = list(result_dict.keys())

    tool_index = {k: i for i, k in enumerate(tool_order)}

    labels = {"classes": tool_order}

    for tool_name, result in result_dict.items():
        for file_name, file_result in result.items():

            if file_name not in labels:
                labels[file_name] = [0.0]*len(tool_order)

            if _is_correct(file_result):
                labels[file_name][
                    tool_index[tool_name]
                ] = 1.0
    return labels


def prune_unsolvable(labels):
    output = {'classes': labels['classes']}

    for k, label in labels.items():
        if k == 'classes':
            continue
        if max(label) == 1:
            output[k] = label

    return output
