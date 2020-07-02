#!/usr/bin/env python3

# Copyright (C) 2007-2017  CAMd
# Please see the accompanying LICENSE file for further information.

from __future__ import print_function
import os
import re
import sys
from setuptools import setup, find_packages
from distutils.command.build_py import build_py as _build_py
from glob import glob
from os.path import join

python_requires = (3, 6)


if sys.version_info < python_requires:
    raise SystemExit('Python 3.6 or later is required!')


install_requires = [
    'numpy>=1.11.3',
    'scipy>=0.18.1',
    'matplotlib>=2.0.0',
]


extras_require = {
    'docs': [
        'sphinx',
        'sphinx_rtd_theme',
        'pillow',
    ],
    'test': [
        'pytest>=3.9.1',
        'pytest-xdist>=1.22.1',
    ]
}


with open('README.rst') as fd:
    long_description = fd.read()

# Get the current version number:
with open('ase/__init__.py') as fd:
    version = re.search("__version__ = '(.*)'", fd.read()).group(1)


package_data = {'ase': ['spacegroup/spacegroup.dat',
                        'collections/*.json',
                        'db/templates/*',
                        'db/static/*'],
                'ase.test': ['pytest.ini',
                             'data/*']}


class build_py(_build_py):
    """Custom distutils command to build translations."""
    def __init__(self, *args, **kwargs):
        _build_py.__init__(self, *args, **kwargs)
        # Keep list of files to appease bdist_rpm.  We have to keep track of
        # all the installed files for no particular reason.
        self.mofiles = []

    def run(self):
        """Compile translation files (requires gettext)."""
        _build_py.run(self)
        msgfmt = 'msgfmt'
        status = os.system(msgfmt + ' -V')
        if status == 0:
            for pofile in sorted(glob('ase/gui/po/*/LC_MESSAGES/ag.po')):
                dirname = join(self.build_lib, os.path.dirname(pofile))
                if not os.path.isdir(dirname):
                    os.makedirs(dirname)
                mofile = join(dirname, 'ag.mo')
                print()
                print('Compile {}'.format(pofile))
                status = os.system('%s -cv %s --output-file=%s 2>&1' %
                                   (msgfmt, pofile, mofile))
                assert status == 0, 'msgfmt failed!'
                self.mofiles.append(mofile)

    def get_outputs(self, *args, **kwargs):
        return _build_py.get_outputs(self, *args, **kwargs) + self.mofiles


setup(name='ase',
      version=version,
      description='Atomic Simulation Environment',
      url='https://wiki.fysik.dtu.dk/ase',
      maintainer='ASE-community',
      maintainer_email='ase-users@listserv.fysik.dtu.dk',
      license='LGPLv2.1+',
      platforms=['unix'],
      packages=find_packages(),
      install_requires=install_requires,
      extras_require=extras_require,
      package_data=package_data,
      entry_points={'console_scripts': ['ase=ase.cli.main:main',
                                        'ase-db=ase.cli.main:old',
                                        'ase-gui=ase.cli.main:old',
                                        'ase-run=ase.cli.main:old',
                                        'ase-info=ase.cli.main:old',
                                        'ase-build=ase.cli.main:old']},
      long_description=long_description,
      cmdclass={'build_py': build_py},
      classifiers=[
          'Development Status :: 6 - Mature',
          'License :: OSI Approved :: '
          'GNU Lesser General Public License v2 or later (LGPLv2+)',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering :: Physics'])
