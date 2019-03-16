from __future__ import print_function, division
import pandas as pd
import random
from pdb import set_trace

TAXI_CAB = 1729
OVERLAP_RANGE = range(0, 101, 25)
random.seed(TAXI_CAB)


def _effectiveness(dframe, thresh_min, thresh_max):
    overlap = dframe['Overlap']
    heeded = dframe['Heeded']

    a, b, c, d = 0, 0, 0, 0

    for over, heed in zip(overlap, heeded):
        if thresh_min < over <= thresh_max and heed >= 0:
            a += 1
        if thresh_min <= over <= thresh_max and heed < 0:
            b += 1
        if over == 0 and heed < 0:
            c += 1
        if over == 0 and heed >= 0:
            d += 1

    return a, b, c, d


def measure_overlap(test, new, validation):
    results = pd.DataFrame()

    "Find modules that appear both in test and validation datasets"
    common_modules = pd.merge(
        test, validation, how='inner', on=['Name'])['Name']

    "Intitialize variables to hold information"
    improve_heeded = []
    overlap = []

    for module_name in common_modules:
        "For every common class, do the following ... "
        same = 0  # Keep track of features that follow XTREE's recommendations
        "Metric values of classes in the test set"
        test_value = test.loc[test['Name'] == module_name]
        "Metric values of classes in the XTREE's planned changes"
        plan_value = new.loc[new['Name'] == module_name]
        "Actual metric values the developer's changes yielded"
        validate_value = validation.loc[validation['Name'] == module_name]

        for col in test_value.columns[1:-1]:
            "For every metric (except the class name and the bugs), do the following ... "
            try:
                if isinstance(plan_value[col].values[0], str):
                    "If the change recommended by XTREE lie's in a range of values"
                    if eval(plan_value[col].values[0])[0] <= validate_value[col].values[0] <= \
                            eval(plan_value[col].values[0])[1]:
                        "If the actual change lies withing the recommended change, then increment the count of heeded values by 1"
                        same += 1
                if isinstance(plan_value[col].values[0], tuple):
                    "If the change recommended by XTREE lie's in a range of values"
                    if plan_value[col].values[0][0] <= validate_value[col].values[0] <= plan_value[col].values[0][1]:
                        "If the actual change lies withing the recommended change, then increment the count of heeded values by 1"
                        same += 1
                elif plan_value[col].values[0] == validate_value[col].values[0]:
                    "If the XTREE recommends no change and developers didn't change anything, that also counts as an overlap"
                    same += 1

            except IndexError:
                "Catch instances where classes don't match"
                pass

        "There are 20 metrics, so, find % of overlap for the class."
        overlap.append(int(same / 20 * 100))
        "Find the change in the number of bugs between the test version and the validation version for that class."
        heeded = test_value['<bug'].values[0] - \
            validate_value['<bug'].values[0]

        improve_heeded.append(heeded)
    
    "Save the results ... "
    validation_common = validation.loc[validation["Name"].isin(common_modules)]

    results['Overlap'] = overlap
    results['Heeded'] = improve_heeded

    return results, [_effectiveness(results, thresh_min=lo, thresh_max=hi) for lo, hi in zip(OVERLAP_RANGE[:-1], OVERLAP_RANGE[1:])]
    # return [tuple(map(lambda x: int(100*x/len(validation_common['<bug'].tolist())), _effectiveness(results, thresh_min=lo, thresh_max=hi))) for lo, hi in zip(OVERLAP_RANGE[:-1], OVERLAP_RANGE[1:])]


def reshape(res_xtree, res_alves, res_shatw, res_olive):
    bugs_increased = []
    bugs_decreased = []

    for thresh, every_res_xtree, every_res_alves, every_res_shatw, every_res_olive in zip(OVERLAP_RANGE[1:], res_xtree, res_alves, res_shatw, res_olive):
        # -- Decrease --
        bugs_decreased.append([thresh, every_res_xtree[0], 'XTREE'])
        bugs_decreased.append([thresh, every_res_alves[0], 'ALVES'])
        bugs_decreased.append([thresh, every_res_shatw[0], 'SHATW'])
        bugs_decreased.append([thresh, every_res_olive[0], 'OLIVE'])

        # -- Increase --
        bugs_increased.append([thresh, every_res_xtree[1], 'XTREE'])
        bugs_increased.append([thresh, every_res_alves[1], 'ALVES'])
        bugs_increased.append([thresh, every_res_shatw[1], 'SHATW'])
        bugs_increased.append([thresh, every_res_olive[1], 'OLIVE'])

    bugs_decreased = pd.DataFrame(bugs_decreased, columns=[
                                  "Overlap", "Num", "Method"])
    bugs_increased = pd.DataFrame(bugs_increased, columns=[
                                  "Overlap", "Num", "Method"])

    return bugs_decreased, bugs_increased


def reshape_overlap(res_xtree, res_alves, res_shatw, res_olive):
    counts = []

    def count_helper(dframe, lo, hi):
        range_hi = dframe['Overlap'].loc[dframe['Overlap'] >= lo].count()
        range_lo = dframe['Overlap'].loc[dframe['Overlap'] > hi].count()
        return range_hi - range_lo

    for thresh_min, thresh_max in zip(OVERLAP_RANGE[:-1], OVERLAP_RANGE[1:]):
        # -- XTREE --
        counts.append([thresh_max, count_helper(res_xtree, thresh_min, thresh_max), 'XTREE'])
        
        # -- ALVES --
        counts.append([thresh_max, count_helper(res_alves, thresh_min, thresh_max), 'ALVES'])
        
        # -- SHATW --
        counts.append([thresh_max, count_helper(res_shatw, thresh_min, thresh_max), 'SHATW'])
        
        # -- OLIVE --
        counts.append([thresh_max, count_helper(res_olive, thresh_min, thresh_max), 'OLIVE'])
        
    counts = pd.DataFrame(counts, columns=["Overlap", "Num", "Method"])

    return counts
