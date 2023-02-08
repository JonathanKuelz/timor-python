#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 17.05.22
from abc import ABC, abstractmethod
from collections.abc import Iterator

from timor.Module import ModuleAssembly, ModulesDB


class AssemblyIterator(ABC, Iterator):
    """Abstract base class for a set of iterators that yield module assemblies.

    This is a base class for assembly iterators. An assembly iterator offers an iterator behavior that returns
    valid module assemblies (=robots) based on a set of rules (generator-specific + global rules) and the database.
    """

    def __init__(self, db: ModulesDB):
        """
        :param db: Module assemblies returned by __next__ will be based on this DB instance.
        """
        self._db: ModulesDB = db
        self.reset()

    def __iter__(self):
        """Iterators are self-iterable: https://peps.python.org/pep-0234/"""
        return self

    def __len__(self):
        """Subclasses that can determine the length without iterating through every element of self should do so"""
        return sum(1 for _ in self)  # Fastest method if there is no easy way of calculating the elements in self

    @abstractmethod
    def __next__(self) -> ModuleAssembly:
        """Implementing next defines how assemblies are built by the iterator"""

    @property
    def db(self) -> ModulesDB:
        """Wrapper property to avoid resetting the db"""
        return self._db

    @property
    @abstractmethod
    def state(self) -> any:
        """The state should suffice to deterministically determine the next element returned by self.__next__"""

    @abstractmethod
    def reset(self) -> None:
        """Should reset the iterator to the initial state"""
