class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        # Start with a very high number so the first price becomes the new min
        min_price = float('inf') 
        max_profit = 0
        
        for price in prices:
            # Update the lowest price found so far
            if price < min_price:
                min_price = price
            
            # Calculate potential profit if we sold today
            current_profit = price - min_price
            
            # Update max_profit if today's profit is better
            if current_profit > max_profit:
                max_profit = current_profit
                
        return max_profit