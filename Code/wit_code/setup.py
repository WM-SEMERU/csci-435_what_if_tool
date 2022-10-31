'''
https://betterscientificsoftware.github.io/python-for-hpc/tutorials/python-pypi-packaging/#creating-a-python-package
'''
from setuptools import setup

setup(
    name='wit_code',
    version='0.0.4',
    description='What-if-tool Code. A Visual Tool for Understanding Machine Learning Models for Software Engineering',
    url='https://github.com/WM-SEMERU/csci-435_what_if_tool',
    author='Ignat Miagkov',
    author_email='iamiagkov@wm.edu',
    license='BSD 2-clause',
    packages=['wit_code'],
    install_requires=['typing',
                      'pandas',
                      'numpy',
                      'jupyter_dash',
                      'plotly',
                      'dash',
                      'huggingface',
                      'transformers'
                      ],
    # https://pypi.org/classifiers/
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.9',
    ],
)
