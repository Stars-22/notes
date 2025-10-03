#!/usr/bin/env python3
import sys
import re
from datetime import datetime

def update_lastmod(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 定义匹配字段的正则表达式
        pattern = r'(\*最后更新:[^\n]*\*)'
        now = datetime.now()
        current_time = f"{now.year}年{now.month}月{now.day}日 {now.hour}:{now.minute:02d}"

        replacement = r'*最后更新: ' + current_time + r'*'

        new_content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
        if count > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Successfully updated lastmod field in {file_path}.")
        else:
            print(f"No updatable lastmod field found in {file_path}.")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        update_lastmod(sys.argv[1])
    else:
        print("Error: Please provide a file path as an argument.")