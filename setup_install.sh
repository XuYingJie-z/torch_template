#!/usr/bin/bash

## 第二步：在当前目录下打包
# python setup.py bdist_egg
python setup.py bdist_wheel
## 第三步：安装
# python setup.py install
pip install --force-reinstall  ./dist/torch_template-0.1.1-py3-none-any.whl
