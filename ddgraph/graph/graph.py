from typing import List, Tuple


class UserItemGraph:
    # item offset is the adj_list length
    adj_list: List[List[Tuple[int, int]]]
    relationships: List[str]
    user_entities: List[str]
    item_entities: List[str]
