from setuptools import setup, find_packages

setup(
    name='micronn',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
    ],
    author='Shaan Chopra',
    description='A minimal neural network library built from scratch',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/freebirdyeah/microNN',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.6',
)
