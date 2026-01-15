import requests
import re
import json
from datetime import datetime
from bs4 import BeautifulSoup
import os

tags_map = {
    1 : "模拟",
    2 : "字符串",
    3 : "动态规划 DP",
    4 : "搜索",
    5 : "数学",
    6 : "图论",
    7 : "贪心",
    8 : "计算几何",
    9 : "暴力数据结构",
    10 : "高精度",
    11 : "树形数据结构",
    12 : "递推",
    13 : "博弈论",
    41 : "莫队",
    42 : "线段树",
    43 : "倍增",
    44 : "线性数据结构",
    45 : "二分",
    47 : "并查集",
    49 : "点分治",
    50 : "平衡树",
    51 : "堆",
    53 : "树状数组",
    54 : "递归",
    55 : "树上启发式合并",
    56 : "单调队列",
    108 : "O2优化",
}

difficulty_map = ["暂无评定", "入门", "普及-", "普及/提高−", "普及+/提高", "提高+/省选−", "省选/NOI−", "NOI/NOI+/CTSC"]

def get_info(url):
    tags = []

    url = '/'.join(url.split('/')[:5]) + '/'
    headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0' }
    
    response = requests.get(url, headers=headers)
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.find('h1').get_text().strip()

    script = soup.find_all('script')[1]
    json_content = script.string.strip()
    data = json.loads(json_content)['data']['problem']
    difficulty = difficulty_map[data['difficulty']]
    for tag in data['tags']:
        if tag in tags_map:
            tags.append(tags_map[tag])
        else:
            tags.append(f"{tag}'未知'")

    tags_text = '`' + '` `'.join(tags) + '`'

    # 返回格式化的问题信息，而不是直接修改全局变量
    return f"""

## 洛谷：{title}
**链接**：{url}  
**相关标签**：{tags_text}  
**难度**：`{difficulty}`
### 方法1：``
- **时间复杂度**：$O()$
- **空间复杂度**：$O()$
```cpp

```"""

# 使用示例
current_date = datetime.now().strftime("%y.%m.%d")
output = f"""# {current_date} 算法刷题笔记"""

url = input('请输入洛谷题目的url：')
while url:
    output += get_info(url)
    url = input('请输入洛谷题目的url：')

print(output)