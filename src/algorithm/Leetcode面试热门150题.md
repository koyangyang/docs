---
title: Leetcode面试热门150题
category: LeetCode
tag:
  - LeetCode
  - 算法
---
## 字符串/数组
### 罗马数字
```python
# 1.整数转罗马数字，直接打表
roman_map = [
            (1000, 'M'),
            (900, 'CM'),
            (500, 'D'),
            (400, 'CD'),
            (100, 'C'),
            (90, 'XC'),
            (50, 'L'),
            (40, 'XL'),
            (10, 'X'),
            (9, 'IX'),
            (5, 'V'),
            (4, 'IV'),
            (1, 'I')
        ]
# 2.罗马数字转整数
roman_map = {
            'I': 1,
            'V': 5,
            'X': 10,
            'L': 50,
            'C': 100,
            'D': 500,
            'M': 1000
        }
        total = 0
        prev_value = 0 # 记录上一个字符
        for char in s:
            value = roman_map[char]
            total += value
            if value > prev_value:# 发现大的在小的左边
                total -= 2 * prev_value
            prev_value = value
        return total
```

### 字符串公共前缀
```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if len(strs) == 0:
            return ""
        s = strs[0] # 假设第一个字符串整体为公共前缀
        for str in strs:
            while not str.startswith(s):
                s = s[:-1] # 如果不是，则缩短字符串继续匹配
                if len(s) == 0:
                    return ""
        return s
```

## 区间
```python
points.sort(key=lambda x: x[0]) # 按左边界排序

points.sort(key=lambda x: x[1]) # 按右边界排序
```

## 滑动窗口
### [209. 长度最小的子数组](https://leetcode.cn/problems/minimum-size-subarray-sum/description/?envType=study-plan-v2&envId=top-interview-150)
> 如果当前窗口的和大于等于target，则更新res，并移动左指针
>

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        i,j = 0,0
        res = float('inf')
        sum = 0
        while j < len(nums):
            sum += nums[j]
            while sum >= target: # 如果当前窗口的和大于等于target，则更新res，并移动左指针
                res = min(res,j-i+1) # 保存当前窗口的长度
                sum -= nums[i] # 移动左指针，减去左指针指向的元素
                i += 1 # 移动左指针
            j += 1 # 移动右指针
        return res if res != float('inf') else 0
```

### [3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/?envType=study-plan-v2&envId=top-interview-150)
>     1. 用一个字典记录每个字符的索引位置
>
>     2. 用两个指针记录当前子串的起始位置和结束位置
>
>     3. 遍历字符串，如果当前字符在字典中，说明有重复字符，更新起始位置
>
>     4. 更新字典中字符的索引位置
>
>     5. 更新最大长度
>
>     6. 返回最大长度
>

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        if n == 1:
            return 1
        char_index = {}
        start = 0
        max_len = 0
        for i in range(n):
            if s[i] in char_index:
                start = max(char_index[s[i]] + 1, start) # 
            char_index[s[i]] = i
            max_len = max(max_len, i - start + 1)
        return max_len
```

## 双指针
### [167. 两数之和 II - 输入有序数组](https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted/description/?envType=study-plan-v2&envId=top-interview-150)
> 利用数组有序的特点，
>
> + 如果 `numbers[i] + numbers[j]` 小于目标值，则移动左指针 `i`，目的是寻找更大的数以增加和。
> + 如果 `numbers[i] + numbers[j]` 大于目标值，则移动右指针 `j`，目的是寻找更小的数以减小和。
>

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        i,j = 0,len(numbers)-1
        while i < j:
            if numbers[i] + numbers[j] == target:
                return [i+1,j+1]
            elif numbers[i] + numbers[j] < target:
                i += 1
            else:
                j -= 1
        return []
```

### [11. 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/description/?envType=study-plan-v2&envId=top-interview-150)
> **移动较短的柱子**：为了寻找更高的容器（从而可能获得更大的容积），**移动较短**的那根柱子的指针（如果左边柱子较短，则将左指针向右移动；反之，则将右指针向左移动）。这是因为**容器的容积取决于较短的那根柱子，只有提高较短部分的高度，才能使容器的容积有潜在上升的可能**。
>

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        i,j = 0,len(height)-1
        res = 0
        while i < j:
            res = max(res,min(height[i],height[j])*(j-i))
            if height[i] < height[j]:
                i += 1
            else:
                j -= 1
        return res
```

### [15. 三数之和](https://leetcode.cn/problems/3sum/?envType=study-plan-v2&envId=top-interview-150)
> **排序！！！ **    
遍历数组中的每个元素，将每个位置的元素作为第一个固定数。  
>
> + 如果当前固定数大于 0，由于数组有序，后续数字都将大于0，因此无法组成和为0的三元组，可以提前中止遍历。
> + 如果和前一个固定数相同，则跳过该元素以避免重复的结果。
>
>    对于每个固定数，使用双指针（left 和 right）在剩下的数组中寻找另两个数字，使得三个数的和为0。
>

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        for i in range(len(nums)):
            if nums[i] > 0:
                break
            if i > 0 and nums[i] == nums[i-1]:
                continue
            left,right = i+1,len(nums)-1
            while left < right:
                if nums[left] + nums[right] + nums[i] == 0:
                    res.append([nums[i],nums[left],nums[right]])
                    while left < right and nums[left] == nums[left+1]:
                        left += 1
                    while left < right and nums[right] == nums[right-1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif nums[left] + nums[right] + nums[i] < 0:
                    left += 1
                else:
                    right -= 1
        return res
```

## 矩阵
### [36. 有效的数独](https://leetcode.cn/problems/valid-sudoku/description/?envType=study-plan-v2&envId=top-interview-150)
> ![](https://disk.csuer.us.kg/1739585429114-c1893299-f3d6-4123-bb59-09a60d02cb78.webp)
>

```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        row = [set() for _ in range(9)]
        col = [set() for _ in range(9)]
        block = [set() for _ in range(9)] # 每个3x3小块 去重
        for i in range(9):
            for j in range(9):
                num = board[i][j]
                if num == '.':
                    continue
                if num in row[i] or num in col[j] or num in block[i // 3 * 3 + j // 3]:
                    return False # block_index索引 i // 3 * 3 + j // 3
                row[i].add(num)
                col[j].add(num)
                block[i // 3 * 3 + j // 3].add(num)
        return True
```

### [54. 螺旋矩阵](https://leetcode.cn/problems/spiral-matrix/description/?envType=study-plan-v2&envId=top-interview-150)
> 1. 初始化矩阵边界 (top, bottom, left, right) 来确定当前遍历区域。
> 2.  顺时针遍历当前区域的四条边：首先遍历上边和右边，当剩余区域至少为2行2列时，再遍历下边和左边以避免重复。
> 3. 每遍历一圈后更新边界，继续遍历下一圈，直到所有元素都被遍历。
>



```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix:
            return []

        res: List[int] = []  # 用于存储遍历的结果
        m: int = len(matrix)
        n: int = len(matrix[0])
        left: int = 0
        right: int = n - 1
        top: int = 0
        bottom: int = m - 1

        # 当左边界不超过右边界，且上边界不超过下边界时，继续遍历
        while left <= right and top <= bottom:
            # 遍历上边：从左到右
            for i in range(left, right + 1):
                res.append(matrix[top][i])
            # 遍历右边：从上到下
            for i in range(top + 1, bottom + 1):
                res.append(matrix[i][right])
            # 只有当当前区域至少含有两行两列时才遍历下边和左边，
            # 以防止交叉重复读取已经添加的那一行或那一列
            if left < right and top < bottom:
                # 遍历下边：从右到左
                for i in range(right - 1, left, -1):
                    res.append(matrix[bottom][i])
                # 遍历左边：从下到上
                for i in range(bottom, top, -1):
                    res.append(matrix[i][left])
            # 更新边界，进入下一层螺旋圈
            left += 1
            right -= 1
            top += 1
            bottom -= 1

        return res
```

### [顺时针90°旋转矩阵](https://leetcode.cn/problems/rotate-image/description/?envType=study-plan-v2&envId=top-interview-150)
```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix)
        # 先上下翻转
        for i in range(n // 2):
            matrix[i], matrix[n - i - 1] = matrix[n - i - 1], matrix[i]
        # 再按对角线翻转
        for i in range(n):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        return matrix
```

### [矩阵置0](https://leetcode.cn/problems/set-matrix-zeroes/description/?envType=study-plan-v2&envId=top-interview-150)
> 优化：先遍历矩阵，如何某个元素为0，则将对应行和列的第一个元素置为0，最后再次遍历矩阵，将对应行和列置为0
>

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        m = len(matrix)
        n = len(matrix[0])
        row = False # 判断第一行是否有0
        col = False # 判断第一列是否有0
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    if i == 0:
                        row = True
                    if j == 0:
                        col = True
                    matrix[i][0] = matrix[0][j] = 0
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        if row:
            for j in range(n):
                matrix[0][j] = 0
        if col:
            for i in range(m):
                matrix[i][0] = 0
        return matrix
```

### [289. 生命游戏](https://leetcode.cn/problems/game-of-life/?envType=study-plan-v2&envId=top-interview-150)
>         该方法通过位运算在原地标记出每个细胞下一个时刻的状态，而不使用额外内存。
>
>         每个细胞的整数的最低位代表当前状态（0表示死亡，1表示活细胞），
>
>         而第二位用于存储下一时刻的状态：
>
>           - 对于每个细胞，通过遍历其 3x3 邻域（包括自身），统计当前活细胞数量（仅利用最低位）。
>
>           - 根据生命游戏规则：
>
>               * 对于死细胞，当且仅当恰有 3 个活细胞时复活。
>
>               * 对于活细胞，统计时自身也被计入，因此需满足周围有2个或3个活邻居（即 count==3 或 count==4）。
>
>           - 为避免干扰当前状态统计，通过将下一状态存储在第二位实现状态并存。
>
>         最后，通过右移一位将更新后的状态设定为当前状态，实现原地更新。
>

```python
from typing import List

class Solution:
    def gameOfLife(self, board: List[List[int]]) -> List[List[int]]:
        m: int = len(board)
        n: int = len(board[0])
        # 第一轮遍历：计算每个细胞周边的活邻居数，并在第二位标记出下一时刻的状态
        for i in range(m):
            for j in range(n):
                count: int = 0
                # 遍历 (i, j) 周围的邻域（3x3 区域），并只统计当前状态（最低位），避免受到之前标记的影响
                for x in range(max(0, i - 1), min(m, i + 2)):
                    for y in range(max(0, j - 1), min(n, j + 2)):
                        count += board[x][y] & 1  # & 1 提取当前状态
                # 判断是否满足复活或存活条件：
                # 对于死细胞：(board[i][j] == 0) 当且仅当恰有3个活细胞（count == 3）复活；
                # 对于活细胞：(board[i][j] == 1) 则 count 包含了细胞自身，
                #     当周围有2个活邻居时 (count == 3) 或周围有3个活邻居时 (count - 1 == 3) 状态保持为活。
                if count == 3 or count - board[i][j] == 3:
                    board[i][j] |= 2  # 将第二位设为1，标记该细胞下一个时刻为活细胞，2的二进制为10
        # 第二轮遍历：更新板上的所有细胞状态，将下一状态（第二位）转化为当前状态
        for i in range(m):
            for j in range(n):
                board[i][j] >>= 1  # 右移一位，完成状态更新
        return board
```

## 哈希表
### [49. 字母异位词分组](https://leetcode.cn/problems/group-anagrams/)
> 将每个字符串排序后作为 key
>

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        hash_map = {}
        for s in strs:
            # 将字符串排序后作为 key
            key = ''.join(sorted(s))
            if key in hash_map:
                hash_map[key].append(s)
            else:
                hash_map[key] = [s]
        return list(hash_map.values()) 
```

### 最长连续串
>  从小到大开始便利（从起点遍历）
>

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        num_set = set(nums)
        max_length = 0
        for num in num_set:
            # 只有当当前数字是序列的起点时（即没有前驱数字）
            if (num - 1) not in num_set:
                current_num = num
                current_length = 1
                # 继续查找后续连续数字
                while (current_num + 1) in num_set:
                    current_num += 1
                    current_length += 1
                max_length = max(max_length, current_length)
        return max_length
```

## 回溯
> 排列组合问题适用
>

### [77. 组合](https://leetcode.cn/problems/combinations/)
```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []
        def backtrack(start, path):
            if len(path) == k:
                res.append(path[:]) # path[:]注意传值调用
                return
            for i in range(start, n+1):
                path.append(i)
                backtrack(i+1, path)
                path.pop() # 回溯
        backtrack(1, [])
        return res
```

### [22. 括号生成](https://leetcode.cn/problems/generate-parentheses/)
从一个空字符串开始，不断地添加 '(' 或 ')' 来构造可能的括号序列。

约束条件

1. 当还有左括号可以用时（即 left > 0），递归地尝试在当前序列后添加左括号，并减少可用的左括号数量。
2. 只有当右括号的数量大于左括号的数量时（right > left），才允许添加右括号，这样可以保证括号序列在任何前缀都是合法的，即不会出现右括号先出现的情况。

递归终止

1. 当没有剩余右括号可用时（即 right == 0），说明已经构造出一个合法的括号序列，此时将该序列添加到结果列表中。

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        def generate(p, left, right, parens=[]):
            if left:
                generate(p + '(', left - 1, right)
            if right > left:
                generate(p + ')', left, right - 1)
            if not right:
                parens += p,
            return parens
        return generate('', n, n)
```

### 迷宫搜索问题
[**<font style="background-color:rgb(240, 240, 240);">79. 单词搜索</font>**](https://leetcode.cn/problems/word-search/)

#### 递归法
```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        m,n = len(board), len(board[0])
        def dfs(i,j,k):
            if not 0<=i<m or not 0<=j<n or board[i][j] != word[k]:
                return False
            if k == len(word)-1:
                return True
            tmp, board[i][j] = board[i][j], '/' # 防止重复访问
            res = dfs(i+1,j,k+1) or dfs(i-1,j,k+1) or dfs(i,j+1,k+1) or dfs(i,j-1,k+1)
            board[i][j] = tmp # 回溯
            return res
        for i in range(m):
            for j in range(n):
                if dfs(i,j,0):
                    return True
        return False
```

#### 迭代法
```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        if not board or not board[0]:
            return False
        
        m, n = len(board), len(board[0])
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
        for i in range(m):
            for j in range(n):
                if board[i][j] == word[0]:
                    # 状态为 (当前行, 当前列, 当前已匹配字符下标, 已访问坐标集合)
                    stack = [(i, j, 0, {(i, j)})]
                    while stack:
                        x, y, index, visited = stack.pop()
                        # 如果匹配到最后一个字符，返回True
                        if index == len(word) - 1:
                            return True
                        for dx, dy in directions:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < m and 0 <= ny < n and (nx, ny) not in visited:
                                if board[nx][ny] == word[index + 1]:
                                    # 将新的状态加入栈中，这里需要拷贝 visited 集合
                                    new_visited = visited.copy()
                                    new_visited.add((nx, ny))
                                    stack.append((nx, ny, index + 1, new_visited))
        return False
```

## <font style="color:rgb(26, 26, 26);">Kadane 算法</font>
### [918. 环形子数组的最大和](https://leetcode.cn/problems/maximum-sum-circular-subarray/)
![](https://disk.csuer.us.kg/1739683113069-8174cc7e-d841-44c7-9064-f7b7218e030b.webp)

```python
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        max_s = -inf  # 最大子数组和，不能为空
        min_s = 0     # 最小子数组和，可以为空
        max_f = min_f = 0 # 遍历数组累加
        for x in nums:
            # 以 nums[i-1] 结尾的子数组选或不选（取 max）+ x = 以 x 结尾的最大子数组和
            max_f = max(max_f, 0) + x
            max_s = max(max_s, max_f)
            # 以 nums[i-1] 结尾的子数组选或不选（取 min）+ x = 以 x 结尾的最小子数组和
            min_f = min(min_f, 0) + x
            min_s = min(min_s, min_f)
        if sum(nums) == min_s:
            return max_s
        return max(max_s, sum(nums) - min_s)
```

## 二分
### [35. 二分查找](https://leetcode.cn/problems/search-insert-position/)
```python
    def find_left(self, nums, target):
        """
        二分查找左边界
        """
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid
        return left if nums[left] == target else -1
    
    def find_right(self, nums, target):
        """
        二分查找右边界
        """
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right + 1) // 2
            if nums[mid] > target:
                right = mid - 1
            else:
                left = mid
        return left if nums[left] == target else -1
```

### [74. 搜索二维矩阵](https://leetcode.cn/problems/search-a-2d-matrix/)
> 先按在哪一行，index = right # 在right那一层继续搜索
>

### [162. 寻找峰值](https://leetcode.cn/problems/find-peak-element/)
> <font style="color:rgb(38, 38, 38);background-color:rgb(240, 240, 240);">保证每次二分出来的区间的[i, j]，满足条件nums[i - 1] < nums[i]且nums[j] < nums[j + 1]即可</font>
>

```python
        while left < right: # 不带等号
            mid = (left + right) // 2
            if nums[mid] > nums[mid + 1]:
                right = mid
            else:
                left = mid + 1
        return left
```

### [33. 搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/)
> **总有一半是有序的！！！**
>
> 在常规二分查找的时候查看当前 mid 为分割位置分割出来的两个部分 [l, mid] 和 [mid + 1, r] 哪个部分是有序的，并根据有序的那部分判断出 target 在不在这个部分：
>
> 如果 [l, mid - 1] 是有序数组，且 target 的大小满足 [nums[l],nums[mid])，则我们应该将搜索范围缩小至 [l, mid - 1]，否则在 [mid + 1, r] 中寻找。
>
> 如果 [mid, r] 是有序数组，且 target 的大小满足 (nums[mid+1],nums[r]]，则我们应该将搜索范围缩小至 [mid + 1, r]，否则在 [l, mid - 1] 中寻找。
>
> ![](https://disk.csuer.us.kg/1739757900553-d73e254c-6388-4dc5-bf18-38a564472c72.webp)
>

```python
while l <= r: # 带等号
    mid = (l + r) // 2
    if nums[mid] == target:
        return mid
    if nums[0] <= nums[mid]:
        if nums[0] <= target < nums[mid]:
            r = mid - 1
        else:
            l = mid + 1
    else:
        if nums[mid] < target <= nums[n - 1]:
            l = mid + 1
        else:
            r = mid - 1
return -1
```

### [153. 寻找旋转排序数组中的最小值](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/)
```python
while left < right: # 没有等号
    mid = (left + right) // 2
    if nums[mid] < nums[right]:
        right = mid
    else:
        left = mid + 1
return nums[left]
```

## 链表
### [146. LRU 缓存](https://leetcode.cn/problems/lru-cache/)
> 双向链表实现
>

## 动态规划
### [198. 打家劫舍](https://leetcode.cn/problems/house-robber/)
> dp[i]表示前i个房子能偷到的最大金额
>

```python
dp[0] = nums[0]
dp[1] = max(nums[0], nums[1])
for i in range(2, len(nums)):
     dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
return dp[-1]
```

### [139. 单词拆分](https://leetcode.cn/problems/word-break/)
> dp[i] 表示字符串 s 的前 i 个字符能否拆分成 wordDict 中的单词
>

```python
dp[0] = True
if dp[j] and s[j:i] in wordDict: # 前j个可以找到，并且j到i也可以找到，则前i个都可以找到
    dp[i] = True
    break
return dp[n]
```

### [322. 零钱兑换](https://leetcode.cn/problems/coin-change/)
> dp[i] 表示凑成金额 i 所需的最少硬币数量
>

```python
dp = [float('inf')] * (amount+1)
dp[0] = 0
for i in range(1, amount+1):
     for coin in coins:
         if i - coin >= 0:
              dp[i] = min(dp[i], dp[i-coin] + 1)
return dp[amount] if dp[amount] != float('inf') else -1
```

