from dataclasses import dataclass
import datetime
import random
import time
import unittest

import numpy as np

from timor.utilities.dtypes import EternalDict, LimitedSizeMap, SingletonMeta, SingleSet, TypedHeader, timeout
import timor.utilities.errors as err


class CustomTypeUnitTests(unittest.TestCase):
    """Tests dtypes in utilities."""

    def setUp(self) -> None:
        # Fix random seeds to have deterministic tests
        random.seed(123)
        np.random.seed(123)

    def test_EternalDict(self):
        d = EternalDict()
        d['first'] = 1
        with self.assertRaises(TypeError):
            d['first'] = 2
        d[2] = 2
        with self.assertRaises(TypeError):
            d[2.] = 2  # lol, python
        d_1 = EternalDict(name='bond', long_name='james bond')
        d_2 = EternalDict(name='bond', profession='agent')
        with self.assertRaises(TypeError):
            d_3 = d_1.update(d_2)
        with self.assertRaises(TypeError):
            del d[2]

    def test_LimitedSizeMap(self):
        with self.assertRaises(ValueError):
            LimitedSizeMap({'a': 1, 'b': 2}, maxsize=-1)
        with self.assertRaises(ValueError):
            LimitedSizeMap(maxsize=0)
        with self.assertRaises(ValueError):
            LimitedSizeMap()

        d = LimitedSizeMap({'a': 1, 'b': 2}, maxsize=1)
        self.assertEqual(len(d), 1)
        d.update({'c': 3, 'd': 4})
        self.assertEqual(len(d), 1)

        d = LimitedSizeMap(maxsize=10)
        for i in range(100):
            d[i] = i
        self.assertEqual(len(d), 10)
        self.assertEqual(list(d), list(range(99, 89, -1)))
        d[90] = 'a new value'
        d['a new key'] = 100
        self.assertEqual(list(d), ['a new key', 90] + list(range(99, 91, -1)))

        d.get(95)  # When accessed, we move an element to the left
        self.assertEqual(95, list(d)[0])

        # Test auto-provided __contains__ and pop
        self.assertTrue(95 in d)
        d.pop(95)
        self.assertFalse(95 in d)

    def test_Singleton(self):
        """Tests the Singleton concept as implemented in SingletonMeta."""
        class Singleton(metaclass=SingletonMeta):
            """This class can only be instantiated once -- every other try will return the initial instance."""

            def __init__(self, val: any):
                self.a = val

        s1 = Singleton(1)
        s2 = Singleton(2)

        self.assertEqual(s2.a, 1)
        self.assertIs(s1, s2)

    def test_SingleSet(self):
        robots = SingleSet()
        robots.add('Panda')
        robots.update(['R2D2', 'Marvin', 'ModRob'])

        names = SingleSet(['Dimitrij', 'Svetlana', 'Marvin'])
        places = SingleSet(['NYC', 'Garching Forschungszentrum', 'Tokyo'])

        original_robots = robots.copy()
        with self.assertRaises(err.UniqueValueError):
            robots.update(names)  # The name "Marvin" is contained in robots and names
        with self.assertRaises(err.UniqueValueError):
            robots.union(names)
        self.assertEqual(robots, original_robots)  # If update fails, it must fail completely

        self.assertIsNone(robots.update(places))  # This however must work
        self.assertEqual(robots, set(robots).union(places))
        self.assertEqual(set(robots), set(robots).union(places))

        self.assertIsInstance(original_robots.union(places), SingleSet)  # Check if union returns the right dtype

        # Finally, a test with random numbers:
        for _ in range(100):
            numbers = np.random.randint(0, 20, (20,))
            original_set = set(numbers)
            if len(numbers) == len(original_set):  # No duplicates
                self.assertEqual(original_set, SingleSet(numbers))
                original_set.pop()
                self.assertNotEqual(original_set, SingleSet(numbers))
            else:
                with self.assertRaises(err.UniqueValueError):
                    SingleSet(numbers)

    def test_StronglyTypedHeader(self):
        @dataclass
        class TestHeader(TypedHeader):
            name: str
            number: int = 2
            date: datetime.date = datetime.date.today()

        header = TestHeader(name='test')
        with self.assertRaises(TypeError):
            header.name = 'new_name'
        with self.assertRaises(TypeError):
            header.__delattr__('name')

        header = TestHeader(name=666, date=datetime.date(1970, 1, 1))
        self.assertIsInstance(header.name, str)
        self.assertEqual(header.name, '666')

        first_date = TestHeader(name='test', date='1970-01-01')
        second_date = TestHeader(name='test', date='1970/1/1')
        self.assertEqual(header.date, first_date.date)
        self.assertEqual(header.date, second_date.date)

    def test_timeout(self):
        x = 0
        with timeout(1):
            time.sleep(1.1)
            x = 1
        self.assertEqual(x, 0)

        with timeout(1):
            time.sleep(0.5)
            x = 2
        self.assertEqual(x, 2)


if __name__ == '__main__':
    unittest.main()
