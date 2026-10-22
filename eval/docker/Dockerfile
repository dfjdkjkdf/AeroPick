FROM debian:trixie-slim
# cppcheck 的 --addon=misra 是个 python 脚本（cppcheck 内部调用），没有 python3
# 装不上这个 addon 就会报错；这个镜像原来一直没装 cppcheck，导致 check_safety()/
# feedback_misra.check() 全程静默失效——cppcheck 命令都找不到，stderr 只有
# "command not found"，正则永远匹配不上，safe@1 一直恒为 True，不代表代码
# 真的没有 MISRA 问题。
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc make libc6-dev gdb cppcheck python3 \
    && rm -rf /var/lib/apt/lists/*
