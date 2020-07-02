"""Check that our tab-completion script has been updated."""
from ase.cli.completion import update, path
from ase.cli.main import commands


def test_complete():
    update(path, commands, test=True)
