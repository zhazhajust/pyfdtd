from setuptools import find_packages, setup

VERSION = '0.0.1'

with open('README.md', 'r', encoding='utf-8') as fp:
    long_description = fp.read()

setup(
    name='pyfdtd',
    version=VERSION,
    author='Hopetree',
    author_email='jiecai@stu.pku.edu.com',
    description='Python FDTD simulation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/Hopetree/django-tctip',
    keywords='fdtd',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numba',
        'numpy',
        'h5py'
    ],
    python_requires='>=3.6',
    classifiers=[
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License'
    ],
)