class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        # using a set to track numbers i've seen
        seen = set()
        for num in nums:
            if num in seen:
                return True
            seen.add(num)
        return False

# alt solution (trick)
# return len(nums) != len(set(nums))