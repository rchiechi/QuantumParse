#!/usr/bin/env python

# Copyright (C) 2007  CAMP
# Please see the accompanying LICENSE file for further information.

import os
import re
import sys
from setuptools import setup, find_packages
from distutils.command.build_py import build_py as _build_py
from glob import glob
from os.path import join


if sys.version_info < (2, 7, 0, 'final', 0):
    raise SystemExit('Python 2.7 or later is required!')


with open('README.rst') as fd:
    long_description = fd.read()

# Get the current version number:
with open('ase/__init__.py') as fd:
    version = re.search("__version__ = '(.*)'", fd.read()).group(1)


package_data = {'ase': ['spacegroup/spacegroup.dat',
                        'collections/*.json',
                        'db/templates/*',
                        'db/static/*']}


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
            for pofile in glob('ase/gui/po/*/LC_MESSAGES/ag.po'):
                dirname = join(self.build_lib, os.path.dirname(pofile))
                if not os.path.isdir(dirname):
                    os.makedirs(dirname)
                mofile = join(dirname, 'ag.mo')
                status = os.system('%s -cv %s --output-file=%s 2>&1' %
                                   (msgfmt, pofile, mofile))
                assert status == 0, 'msgfmt failed!'
                self.mofiles.append(mofile)

    def get_outputs(self, *args, **kwargs):
        return _build_py.get_outputs(self, *args, **kwargs) + self.mofiles


name = 'ase'  # PyPI name

# Linux-distributions may want to change the name:
if 0:
    name = 'python-ase'

setup(name=name,
      version=version,
      description='Atomic Simulation Environment',
      url='https://wiki.fysik.dtu.dk/ase',
      maintainer='ASE-community',
      maintainer_email='ase-developers@listserv.fysik.dtu.dk',
      license='LGPLv2.1+',
      platforms=['unix'],
      packages=find_packages(),
      install_requires=['numpy', 'scipy', 'matplotlib', 'flask'],
      # package_dir=package_dir,
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
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Physics'])
