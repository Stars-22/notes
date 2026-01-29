# 汇编笔记

## 常用网站
- syscall：https://blog.rchapman.org/posts/Linux_System_Call_Table_for_x86_64/

## 指令

- `as asem.s -o asem.o` 汇编
- `ld asem.o -o asem` 链接

## 语法

### 转运
- `mov a, b` 将b的内容移动到a里
- `push a` 将a压入栈
- `pop a` 将栈顶元素给a，并出栈
- 尾部
  - `b` 字节
  - `w` 字（2Byte）
  - `l` 双字（4Byte）
  - `q` 四字（8Byte）

### 运算
- `INC D` D <- D+1
- `ADD S, D` D <- D+S
- `SUB S, D` D <- D-S
- `OR S, D` D <- D|S
- `AND S, D` D <- D&S
- `cmp S2, S1` 比较，基于S1-S2设置条件码
- `test S2, S1` 测试，基于S1&S2设置条件码

### 跳转
- `jump label` 直接跳转
- `jump *operand` 间接跳转
- `je` ==
- `jne` !=
- `js` <0
- `jns` >=0
- `jg` > (有符号)
- `jge` >= (有符号)
- `jl` < (有符号)
- `jle` <= (有符号)
- `ja` > (无符号)
- `jae` >= (无符号)
- `jb` < (无符号)
- `jbe` <= (无符号)

### 条件转运
- `cmov R, S` 满足条件时，将S值复制到R
- `cmove R, S` ==
- `cmovne R, S` !=
- `cmovs R, S` <=0
- `cmovns R, S` >=0

### 函数调用
- `call label` 函数调用，内部操作：
  - 1. call指令的下一条指令地址压入栈
  - 2. label放入%eip
- `call *operand` 间接调用
- `ret` 函数返回，内部操作：pop %eip

### 其他
- `syscall` 调用内核功能
- `lea a, [b]` 计算b表达式的地址，存入a里 
- `.ascii "b"` 将b字符串保存（末尾无\0）
- `.asciz "b"` 将b字符串保存（末尾有\0）
- `.global _start` 将 _start 声明为“全局可见”，使得链接器能将其作为程序的入口点。
- `_start:` 操作系统加载程序后跳转执行的第一个地址
- `//` 单行注释