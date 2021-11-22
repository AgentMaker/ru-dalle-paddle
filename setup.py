# -*- coding: utf-8 -*-
import os
import re
from setuptools import setup


def read(filename):
    with open(os.path.join(os.path.dirname(__file__), filename)) as f:
        file_content = f.read()
    return file_content


def get_requirements():
    requirements = []
    for requirement in read('requirements.txt').splitlines():
        if requirement.startswith('git+') or requirement.startswith('svn+') or requirement.startswith('hg+'):
            parsed_requires = re.findall(r'#egg=([\w\d\.]+)-([\d\.]+)$', requirement)
            if parsed_requires:
                package, version = parsed_requires[0]
                requirements.append(f'{package}=={version}')
            else:
                print('WARNING! For correct matching dependency links need to specify package name and version'
                      'such as <dependency url>#egg=<package_name>-<version>')
        else:
            requirements.append(requirement)
    return requirements


def get_links():
    return [
        requirement for requirement in read('requirements.txt').splitlines()
        if requirement.startswith('git+') or requirement.startswith('svn+') or requirement.startswith('hg+')
    ]


def get_version():
    """ Get version from the package without actually importing it. """
    init = read('rudalle_paddle/__init__.py')
    for line in init.split('\n'):
        if line.startswith('__version__'):
            return eval(line.split('=')[1])


setup(
    name='rudalle_paddle',
    version=get_version(),
    author='SberAI, SberDevices',
    author_email='',
    description='',
    packages=[
        'rudalle_paddle',
        'rudalle_paddle/dalle',
        'rudalle_paddle/realesrgan',
        'rudalle_paddle/ruclip',
        'rudalle_paddle/vae',
        'rudalle_paddle/packages',
        'rudalle_paddle/packages/einops',
        'rudalle_paddle/packages/einops/layers',
        'rudalle_paddle/packages/taming/modules/diffusionmodules',
        'rudalle_paddle/packages/transformers',
    ],
    package_data={'rudalle_paddle/vae': ['*.yml']},
    install_requires=get_requirements(),
    dependency_links=get_links(),
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
)
