#!/bin/bash

set -xe

# From: https://blog.csdn.net/star_xiong/article/details/43529637

# Check the limit of system's core file.
# If it is 0, core file will not be generated.
# e.g.
#    core file size          (blocks, -c) 0
ulimit -a

# Set the length of core file to unlimited
ulimit -c unlimited
# e.g core file size          (blocks, -c) unlimited
ulimit -a
