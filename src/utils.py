from typing import  List

def check_lists_equal(list1: List[str], list2: List[str]) -> bool:
    for elem1 in list1:
        if elem1 not in list2:
            return False
    return True