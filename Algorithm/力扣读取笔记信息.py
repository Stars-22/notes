import requests
import re
import json
from datetime import datetime
from bs4 import BeautifulSoup
import os

def get_info(url):
    # 在函数内部定义变量，避免使用全局变量
    tags = []
    difficulty = ''
    
    url = '/'.join(url.split('/')[:5]) + '/'

    soup = BeautifulSoup(requests.get(url).text, 'html.parser')

    title = soup.find_all('title')[0].text.split(' - 力扣（LeetCode）')[0]
    scripts = soup.find_all('script')
    for script in scripts:
        if script.string and 'topicTags' in script.string:
            match = re.search(r'"topicTags":\s*(\[.*?\])', script.string, re.DOTALL)
            if match:
                try:
                    json_data = json.loads(match.group(1))
                    for j in json_data:
                        tags.append(j['translatedName'])
                except json.JSONDecodeError:
                    pass
        if script.string and 'difficulty' in script.string:
            match = re.search(r'"difficulty":\s*(\".*?\")', script.string, re.DOTALL)
            if match:
                difficulty = match.group(1)

    tags_text = '`' + '` `'.join(tags) + '`'

    if difficulty == '"Easy"':
        difficulty = '简单'
    elif difficulty == '"Medium"':
        difficulty = '中等'
    elif difficulty == '"Hard"':
        difficulty = '困难'
    
    # 返回格式化的问题信息，而不是直接修改全局变量
    return f"""

## 力扣：{title}
**链接**：{url}  
**相关标签**：{tags_text}  
**难度**：`{difficulty}`
### 方法1：``
- **时间复杂度**：$O()$
- **空间复杂度**：$O()$
```cpp

```"""




current_date = datetime.now().strftime("%y.%m.%d")
output = f"""# {current_date} 算法刷题笔记"""

url = input('请输入力扣题目的url：')
while url:
    output += get_info(url)
    url = input('请输入力扣题目的url：')

folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), datetime.now().strftime("%y.%m"))
filepath = os.path.join(folder_path, current_date + '.md')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(output)

print(f"算法笔记已保存到文件: {filepath}")