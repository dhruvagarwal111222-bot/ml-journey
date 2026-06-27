# The isBadVersion API is already defined for you.
# def isBadVersion(version: int) -> bool:

class Solution:
    def firstBadVersion(self, n: int) -> int:
        left, right = 1, n          # no list needed, search directly
        while left < right:         # stops when left == right == answer
            mid = (left + right) // 2
            if isBadVersion(mid):
                right = mid         # mid could be first bad, keep it in range
            else:
                left = mid + 1      # mid is good, first bad is strictly after
        return left
        