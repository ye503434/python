prices = [10, 1, 5, 6, 7, 1]
ans = 0
for i in range(len(prices)):
    buy = prices[i]
    for j in range(i + 1, len(prices)):
        sell = prices[j]
        ans = max(ans, sell - buy)
print(ans)

s = "VIII"
s.replace("dd" , "III")