# -*- coding: utf-8 -*-
#
# Copyright (c) 2017, the cclib development team
#
# This file is part of cclib (http://cclib.github.io) and is distributed under
# the terms of the BSD 3-Clause License.

"""Test scan logfiles in cclib"""

import os
import unittest

import numpy
import cclib

from skip import skipForParser


__filedir__ = os.path.realpath(os.path.dirname(__file__))


class GenericScanTest_optdone_bool(unittest.TestCase):
    """Generic relaxed potential energy surface scan unittest."""

    datatype = cclib.parser.data.ccData_optdone_bool

    def testoptdone(self):
        """Is the optimization finished?"""
        self.assertIsInstance(self.data.optdone, bool)
        self.assertEquals(self.data.optdone, True)

    def testindices(self):
        """Do the indices match the results from geovalues."""
        assert self.data.optdone and numpy.all(self.data.geovalues[-1] <= self.data.geotargets)

    @skipForParser("Jaguar", "Not implemented")
    @skipForParser("ORCA", "Not implemented")
    def testoptstatus(self):
        """Does optstatus contain expected values?"""
        OPT_DONE = self.data.OPT_DONE

        # The input and final coordinates were at a stationary points.
        self.assertEquals(self.data.optstatus[0], OPT_DONE)
        self.assertEquals(self.data.optstatus[-1], OPT_DONE)


class GenericScanTest(unittest.TestCase):
    """Generic relaxed potential energy surface scan unittest."""

    # extra indices
    extra = 0

    def testnumindices(self):
        """Do the number of indices match number of scan points."""
        self.assertEquals(len(self.data.optdone), 12 + self.extra)

    @skipForParser("Jaguar", "Does not work as expected")
    @skipForParser("ORCA", "Does not work as expected")
    def testindices(self):
        """Do the indices match the results from geovalues."""
        indexes = self.data.optdone
        geovalues_from_index = self.data.geovalues[indexes]
        temp = numpy.all(self.data.geovalues <= self.data.geotargets, axis=1)
        geovalues = self.data.geovalues[temp]
        numpy.testing.assert_array_equal(geovalues, geovalues_from_index)

    @skipForParser("Gaussian", "Not working as expected")
    @skipForParser("Jaguar", "Not implemented")
    @skipForParser("ORCA", "Not implemented")
    def testoptstatus(self):
        """Does optstatus contain expected values?"""
        OPT_NEW = self.data.OPT_NEW
        OPT_DONE = self.data.OPT_DONE

        # The input coordinates were at a stationary point.
        self.assertEquals(self.data.optstatus[0], OPT_DONE)

        self.assertEqual(len(self.data.optstatus), len(self.data.optdone))
        for idone in self.data.optdone:
            self.assertEquals(self.data.optstatus[idone], OPT_DONE)
            if idone != len(self.data.optdone) - 1:
                self.assertEquals(self.data.optstatus[idone+1], OPT_NEW)


class GaussianScanTest(GenericScanTest):
    """Customized relaxed potential energy surface scan unittest"""
    extra = 1


class JaguarScanTest(GenericScanTest):
    """Customized relaxed potential energy surface scan unittest"""
    extra = 1


class OrcaScanTest(GenericScanTest):
    """Customized relaxed potential energy surface scan unittest"""
    extra = 1


if __name__=="__main__":

    import sys
    sys.path.append(os.path.join(__filedir__, ".."))

    from test_data import DataSuite
    suite = DataSuite(['Scan'])
    suite.testall()