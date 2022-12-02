from setuptools import setup, find_packages


with open("README.md", "r") as readme_file:
    readme_text = readme_file.read()


setup_args = dict(
    name='tekleo-common-utils-ai',
    version='0.0.0.5',
    description="",
    keywords=[],
    long_description=readme_text,
    long_description_content_type="text/markdown",
    license='Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International Public License',
    packages=find_packages(),
    author="Leo Ertuna",
    author_email="leo.ertuna@gmail.com",
    url="https://github.com/jpleorx/tekleo-common-utils-ai",
    download_url='https://pypi.org/project/tekleo-common-utils-ai/'
)


install_requires = [
    'tekleo-common-message-protocol',
    'tekleo-common-utils',
    'labelme',
    'pycocotools',
    'imgviz',
]


if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)
