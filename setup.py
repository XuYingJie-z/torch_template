from setuptools import setup, find_packages

setup(
    name = 'torch_template',
    version = '0.1.1',
    keywords='torch template',
    description = 'a library for write torch code',
    license = 'MIT License',
    # url = 'https://github.com/Gutier14/CAAFinder',
    author = 'Yingjie Xu',
    author_email = 'xuyingjie1048@foxmail.com',
    packages = find_packages(),
    include_package_data = True,
    platforms = 'any',
    install_requires = [],
)