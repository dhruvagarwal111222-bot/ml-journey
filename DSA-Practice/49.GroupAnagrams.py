from collections import defaultdict
from typing import List

class Solution:
    # Approach 1: Sort each word as key — O(n * k log k) time, O(n * k) space
    def groupAnagrams_sort(self, strs: List[str]) -> List[List[str]]:
        groups = defaultdict(list)
        for word in strs:
            key = ''.join(sorted(word))
            groups[key].append(word)
        return list(groups.values())

    # Approach 2: Character frequency as key — O(n * k) time, O(n * k) space
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        groups = defaultdict(list)
        for word in strs:
            count = [0] * 26
            for char in word:
                count[ord(char) - ord('a')] += 1
            groups[tuple(count)].append(word)
        return list(groups.values())